"""Real CPU tensor checks for bounded and zero-Target-context DFlash."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config
from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.core.dflash import OnlineDFlashModel
from litespec.core.attention_masks import create_dflash_sdpa_mask
from litespec.vlm.baselines import DraftCache


def tiny(mode, device='cpu',width=32):
    cfg=Qwen3Config(vocab_size=128,hidden_size=width,intermediate_size=2*width,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=width//4,
        max_position_embeddings=4096,layer_types=['full_attention']*2,attention_dropout=0.)
    cfg.architectures=['DFlashDraftModel'];cfg.block_size=8;cfg.num_target_layers=32
    cfg.dflash_config=dict(target_layer_ids=[1,8,15,22,29],mask_token_id=127)
    cfg.vlm_config=dict(draft_context=mode);cfg._attn_implementation='sdpa'
    return DFlashDraftModel(cfg).to(device)


class RecentTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_train_loss_gradients_prefix_parity_and_checkpoint_reload(self):
        for mode,window in [('recent64',64),('recent256',256),('recent0',0)]:
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                draft=tiny(mode)
                emb=torch.nn.Embedding(128,32).requires_grad_(False)
                head=torch.nn.Linear(32,128,bias=False).requires_grad_(False)
                online=OnlineDFlashModel(draft,head,emb,127,block_size=8,num_anchors=2,
                    attention_backend='sdpa',selector_loss_alpha=0)
                ids=torch.randint(0,120,(1,300))
                features=torch.randn(1,300,160,requires_grad=True)
                anchors=torch.tensor([[12,280]]);keep=torch.ones_like(anchors,dtype=torch.bool)
                online._sample_anchor_positions=lambda *a,**kw:(anchors,keep)
                _,_,trained=online._forward_draft_blocks(ids,features,torch.ones_like(ids).float())
                for i,a in enumerate(anchors[0].tolist()):
                    query_ids=ids.new_full((1,8),127);query_ids[:,0]=ids[:,a]
                    left=max(0,a-window)
                    context=features[:,left:a] if window else features[:,:0]
                    positions=torch.cat([torch.arange(left,a) if window else torch.empty(0,dtype=torch.long),
                                         torch.arange(a,a+8)])[None]
                    inferred=draft(position_ids=positions,target_hidden=context,noise_embedding=emb(query_ids),is_causal=False)
                    torch.testing.assert_close(trained[:,i*8:(i+1)*8],inferred,atol=2e-6,rtol=2e-5)
                loss=online(input_ids=ids,hidden_states=features,loss_mask=torch.ones_like(ids).float()).loss
                self.assertTrue(torch.isfinite(loss));loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters() if p.requires_grad))
                if window==0:
                    self.assertIsNone(features.grad)
                    self.assertEqual(draft.target_layer_ids,[])
                    self.assertFalse(any(n.startswith(('fc.','hidden_norm.','context_depth.')) for n,_ in draft.named_parameters()))
                else:self.assertGreater(features.grad.abs().sum().item(),0)
                with tempfile.TemporaryDirectory() as path:
                    draft.save_pretrained(path)
                    reloaded=DFlashDraftModel.from_pretrained(path,attn_implementation='sdpa').eval()
                    self.assertEqual(reloaded.context_window,window)
                    torch.testing.assert_close(reloaded(position_ids=positions,target_hidden=context,noise_embedding=emb(query_ids),is_causal=False),inferred)

    def test_recent0_ignores_hidden_and_separates_training_blocks(self):
        torch.manual_seed(8);draft=tiny('recent0').eval()
        query=torch.randn(1,16,32);positions=torch.tensor([[*range(20,28),*range(100,108)]])
        anchors=torch.tensor([[20,100]])
        mask=create_dflash_sdpa_mask(anchors,torch.ones_like(anchors,dtype=torch.bool),0,8,'cpu',context_window=0)
        first=draft(position_ids=positions,noise_embedding=query,target_hidden=torch.randn(1,512,160),attention_mask=mask,is_causal=False)
        second=draft(position_ids=positions,noise_embedding=query,target_hidden=None,attention_mask=mask,is_causal=False)
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        changed=query.clone();changed[:,8:]=torch.randn_like(changed[:,8:])*100
        altered=draft(position_ids=positions,noise_embedding=changed,target_hidden=None,attention_mask=mask,is_causal=False)
        torch.testing.assert_close(first[:,:8],altered[:,:8],atol=0,rtol=0)
        self.assertEqual(mask.shape,(1,1,16,16))
        self.assertTrue(mask[0,0,:8,:8].all());self.assertFalse(mask[0,0,:8,8:].any())
        with self.assertRaisesRegex(ValueError,'no cross-round'):
            draft(position_ids=positions,noise_embedding=query,use_cache=True)

    def test_recent0_cache_never_extracts_hidden_and_keeps_zero_storage(self):
        draft=tiny('recent0').eval()
        emb=torch.nn.Embedding(128,32);head=torch.nn.Linear(32,128,bias=False)
        target=SimpleNamespace(get_input_embeddings=lambda:emb,get_output_embeddings=lambda:head)
        output=SimpleNamespace(logits=torch.zeros(1,1,128))  # deliberately no hidden_states
        with patch.object(DraftCache,'features',side_effect=AssertionError('Target hidden was read')):
            cache=DraftCache(draft,target,output,1000)
            for a in [1000,1003,1005]:
                proposed,counts=cache.propose(draft,target,torch.tensor([[7]]),a)
                self.assertEqual(proposed.shape,(1,7));self.assertEqual(counts['physical_history'],0)
                cache.commit(draft,object(),3)
                self.assertEqual(cache.cache.get_seq_length(),0)
                self.assertEqual(cache.hidden.untyped_storage().nbytes(),0)

    def test_bounded_cache_releases_full_prefix_storage(self):
        for mode,window in [('recent64',64),('recent256',256)]:
            draft=tiny(mode)
            output=SimpleNamespace(hidden_states=tuple(torch.randn(1,500,32) for _ in range(33)))
            cache=DraftCache(draft,None,output,500)
            self.assertEqual(cache.hidden.shape,(1,window,160))
            self.assertEqual(cache.hidden.untyped_storage().nbytes(),cache.hidden.numel()*cache.hidden.element_size())

    def test_real_multimodal_target_reference_decode_for_new_contexts(self):
        from test_vlm import tiny_qwen35,tiny_image_inputs
        from litespec.vlm.baselines import make_draft
        from litespec.vlm.runtime import PreparedExample,generate
        from litespec.vlm.context import visual_positions
        from litespec.vlm.benchmark import RequestPositionState
        processor=SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda:{str(i):i for i in range(127)}))
        for mode in ('recent256','recent64','recent0'):
            with self.subTest(mode=mode):
                torch.manual_seed(7);target=tiny_qwen35().requires_grad_(False)
                draft=make_draft(target,processor,dict(algorithm='dflash',draft_context=mode,
                    num_draft_layers=1,block_size=8,target_model='tiny',context_policy={})).eval()
                inputs=tiny_image_inputs()
                example=PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0],target.config),'image')
                with RequestPositionState(target).activate():
                    result=generate(target,draft,example,max_new_tokens=9,verification='sequential')
                self.assertGreater(len(result['token_ids']),0)
                if mode=='recent0':
                    self.assertTrue(all(r['physical_history']==0 for r in result['context_counts']))

    def test_recent0_static_sequential_path_never_requests_target_hidden(self):
        # The CPU Hub Transformers build lacks the server's transaction extension.
        # The native Qwen cache suffices for sequential verification (no rollback).
        from transformers import DynamicCache
        from transformers.models.qwen3_5 import modeling_qwen3_5
        from test_vlm import tiny_qwen35,tiny_image_inputs
        from litespec.vlm.baselines import make_draft
        from litespec.vlm.runtime import PreparedExample
        from litespec.vlm.context import visual_positions
        from litespec.vlm.static_batch import generate_static_batch
        from litespec.vlm.benchmark import RequestPositionState
        target=tiny_qwen35().requires_grad_(False)
        processor=SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda:{str(i):i for i in range(127)}))
        draft=make_draft(target,processor,dict(algorithm='dflash',draft_context='recent0',num_draft_layers=1,
            block_size=8,target_model='tiny',context_policy={})).eval()
        inputs=tiny_image_inputs();example=PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0],target.config),'image')
        native=getattr(modeling_qwen3_5,'Qwen3_5DynamicCache',None)
        factory=lambda t,c:native(t.config.text_config) if native else DynamicCache(config=t.config)
        original=target.forward;calls=[]
        def guarded(*args,**kwargs):
            calls.append(kwargs)
            self.assertLess(len(calls),32,'decode did not advance after disabling hidden outputs')
            return original(*args,**kwargs)
        with patch('litespec.vlm.static_batch.make_hybrid_cache',side_effect=factory), \
             patch.object(target,'forward',side_effect=guarded) as forward,RequestPositionState(target).activate():
            result=generate_static_batch(target,draft,example,max_new_tokens=9,verification='sequential',
                state_mode='full_target_replay',profile=True)
        self.assertTrue(all(not c.kwargs.get('output_hidden_states',False) for c in forward.call_args_list))
        self.assertGreater(len(result['token_ids']),0)
        self.assertEqual(result['profile']['draft_cache']['projected_kv_positions'],0)
        self.assertEqual(result['profile']['draft_cache']['raw_hidden_storage_bytes'],0)


if __name__=='__main__':unittest.main()
