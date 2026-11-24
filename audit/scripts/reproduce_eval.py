"""Standalone metric reproduction script for audit verification."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

from PIL import Image
from src.inference import RoadVisionSegmentor
from src.metrics import SegmentationEvaluator
from src.constants import get_classes, get_ignore_index

SCENES = [
    'frankfurt_000000_000294','frankfurt_000000_000576',
    'frankfurt_000000_001016','frankfurt_000000_001236','frankfurt_000000_001751'
]
img_dir = 'data/samples/authentic'
lbl_dir = 'data/labels/authentic'
IGNORE = get_ignore_index()

seg = RoadVisionSegmentor(device='cpu')
evaluator = SegmentationEvaluator(num_classes=19, ignore_index=IGNORE, class_names=get_classes())

per_image_results = []
for s in SCENES:
    img_path = os.path.join(img_dir, s+'.png')
    lbl_path = os.path.join(lbl_dir, s+'_label.png')
    
    pil_img = Image.open(img_path).convert('RGB')
    pred_res = seg.predict(pil_img)
    pred_mask = pred_res['pred_mask']
    
    gt_arr = np.array(Image.open(lbl_path))
    if gt_arr.shape != pred_mask.shape:
        gt_pil = Image.fromarray(gt_arr).resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
        gt_arr = np.array(gt_pil)
    
    # Per-image evaluation
    img_eval = SegmentationEvaluator(num_classes=19, ignore_index=IGNORE, class_names=get_classes())
    img_eval.update(pred_mask, gt_arr)
    img_res = img_eval.compute()
    per_image_results.append({'scene': s, 'mIoU': img_res['mIoU'], 'pixel_accuracy': img_res['pixel_accuracy']})
    
    evaluator.update(pred_mask, gt_arr)
    print(f'  Processed {s}: mIoU={img_res["mIoU"]*100:.2f}%  px_acc={img_res["pixel_accuracy"]*100:.2f}%')

result = evaluator.compute()
print('=== AGGREGATE METRICS ===')
print(f'mIoU:            {result["mIoU"]*100:.4f}%')
print(f'pixel_accuracy:  {result["pixel_accuracy"]*100:.4f}%')
print(f'evaluated_classes: {result["evaluated_classes_count"]}')
print(f'total_pixels:    {result["total_pixels_evaluated"]}')
print()
print('Per-class IoU:')
for cls, iou in result['per_class_iou'].items():
    print(f'  {cls:20s}: {iou*100:.2f}%' if not (isinstance(iou, float) and iou != iou) else f'  {cls:20s}: NaN (not present)')

print()
print('Per-image mIoU:')
img_miou_list = [r['mIoU'] for r in per_image_results]
for r in per_image_results:
    print(f'  {r["scene"]}: mIoU={r["mIoU"]*100:.2f}%')
print(f'  Image-level mean mIoU: {np.mean(img_miou_list)*100:.2f}%')

# Save result
out = {
    'reproduced_aggregate_mIoU': result['mIoU'],
    'reproduced_pixel_accuracy': result['pixel_accuracy'],
    'evaluated_classes_count': result['evaluated_classes_count'],
    'total_pixels_evaluated': result['total_pixels_evaluated'],
    'per_image_results': per_image_results,
    'image_level_mean_mIoU': float(np.mean(img_miou_list)),
    'per_class_iou': {k: (v if not (isinstance(v, float) and v != v) else None) for k, v in result['per_class_iou'].items()}
}
with open('audit/evidence/metrics/reproduced_clean_eval.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved to audit/evidence/metrics/reproduced_clean_eval.json')
