"""CPU fixtures for deterministic selection, atomic publication and page inputs."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_multimodal_longctx as prep
from encode_multimodal_longctx import document_example, length_bin


class PreparationTests(unittest.TestCase):
    def test_cap_spreads_questions_and_is_reproducible(self):
        rows=[dict(doc_id=f'{i//4}.pdf',question=str(i)) for i in range(120)]
        a=prep.capped_rows(rows,100,42)
        self.assertEqual(a,prep.capped_rows(rows,100,42))
        self.assertEqual(len(a),100)
        self.assertEqual(len({r['doc_id'] for _,r in a[:30]}),30)
        self.assertEqual(len(prep.capped_rows(rows[:8],100,42)),8)

    def test_atomic_reuse_and_changed_input_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';settings={'seed':42}
            def writer(stage):
                (stage/'asset').write_text('ok')
                return [dict(id='a',question='hello')],{}
            prep.atomic_set(root,settings,writer)
            with patch.object(prep,'sha256',wraps=prep.sha256),redirect_stdout(io.StringIO()):
                prep.atomic_set(root,settings,lambda s:self.fail('must reuse'))
            with self.assertRaises(ValueError):prep.verify(root,{'seed':1})
            (root/'asset').write_text('bad')
            with self.assertRaises(ValueError):prep.verify(root,settings)

    def test_failed_conversion_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data'
            def writer(stage):
                (stage/'partial').write_text('x');raise ValueError('invalid')
            with self.assertRaises(ValueError):prep.atomic_set(root,{},writer)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_video_has_same_ids_across_frame_variants_and_caps_to_available(self):
        import types
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'videos'/'Video-MME'
            (folder/'data').mkdir(parents=True)
            annotation=folder/'videomme/test-00000-of-00001.parquet'
            annotation.parent.mkdir();annotation.write_bytes(b'fixture')
            rows=[]
            for i in range(6):
                (folder/'data'/f'{i}.mp4').write_bytes(b'fake video')
                rows.append(dict(videoID=str(i),duration=['short','medium','long'][i%3]))
            fake=types.ModuleType('pandas')
            fake.read_parquet=lambda p:SimpleNamespace(to_dict=lambda mode:rows)
            args=SimpleNamespace(video_root=root/'videos',output_dir=root/'out',samples=100,seed=42,
                frames=[32,64,128],video_max_pixels=200704)
            with patch.dict(sys.modules,{'pandas':fake}):m=prep.prepare_video(args,'Video-MME')
            self.assertEqual(m['samples'],6)
            sets=[prep.jsonl(root/'out'/'Video-MME'/f'frames_{f}.jsonl') for f in args.frames]
            self.assertEqual([r['id'] for r in sets[0]],[r['id'] for r in sets[2]])
            self.assertTrue(all('answer' not in r for r in sets[0]))

    def test_document_adapter_keeps_all_pages_and_never_inputs_answer(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            paths=[]
            for i in range(3):
                path=Path(temp)/f'{i}.png';Image.new('RGB',(4+i,5),'white').save(path);paths.append(str(path))
            calls=[]
            class Processor:
                def apply_chat_template(self,messages,**kwargs):
                    calls.append(messages)
                    return dict(input_ids=torch.tensor([[1,7,7,7,2]]),image_grid_thw=torch.ones(3,3,dtype=torch.long))
            row=dict(id='doc',images=paths,question='How many?',answer='SECRET')
            with patch('litespec.vlm.context.visual_positions',lambda ids,c:ids==7):
                e=document_example(Processor(),None,row)
            self.assertEqual(e.prefill_length,5)
            content=calls[0][0]['content']
            self.assertEqual([v['image'].width for v in content[:-1]],[4,5,6])
            self.assertEqual(content[-1],dict(type='text',text='How many?'))
            self.assertNotIn('SECRET',str(calls))

    def test_render_all_pdf_pages_and_reuse_across_questions(self):
        try:import fitz
        except ImportError:self.skipTest('PyMuPDF not installed')
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);pdf=root/'sample.pdf'
            with fitz.open() as d:
                for i in range(3):d.new_page(width=100,height=200)
                d.save(pdf)
            dest=root/'out';stage=root/'stage';stage.mkdir()
            selected=[(i,dict(doc_id='sample.pdf',question=f'Q{i}',answer='x',evidence_pages='[2]')) for i in range(2)]
            records,meta=prep.document_records(selected,dest,stage,lambda key:pdf,128)
            self.assertEqual(len(list(stage.rglob('*.jpg'))),3)
            self.assertEqual(records[0]['images'],records[1]['images'])
            self.assertEqual(records[0]['page_numbers'],[1,2,3])
            self.assertEqual(meta['documents']['sample.pdf']['pages'],3)

    def test_bins(self):
        self.assertEqual(length_bin(8192),'le_8192');self.assertEqual(length_bin(90000),'gt_65536')


if __name__=='__main__':unittest.main()
