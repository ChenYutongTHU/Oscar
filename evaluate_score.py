from oscar.utils.caption_evaluate import (evaluate_on_coco_caption)
import json
split='out'
# predict_file = '/data/private/NocapsData/nocaps_base_scst/pred.nocaps_own_tags.val.beam5.max20.odlabels.cbs2.json'
# caption_file = '/data/private/NocapsData/nocaps1k_kara/{}.caption_coco_format.json'.format(split)

predict_file = './caption_output/64_8_epoch60_5e-5/select/checkpoint-52-58671.0/pred.nocaps_{}.{}.beam5.max20.odlabels.cbs2_coco_format.json'.format(split, split)
caption_file = '/data/private/NocapsData/nocaps1k_kara/{}.caption_coco_format.json'.format(split)
assert '.json' in predict_file
evaluate_file = predict_file.replace('.json','_evaluation.json')
evaluate_file_imgscore = predict_file.replace('.json','_evaluation_imgscores.json')
result, img2eval = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file, return_imgscores=True)
with open(evaluate_file_imgscore,'w') as f:
    json.dump(img2eval, f)
print('evaluation result: {}'.format(str(result)))
print('evaluation result saved to {}'.format(evaluate_file))