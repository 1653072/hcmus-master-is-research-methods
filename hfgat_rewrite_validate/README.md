# H-FGAT rewrite for a 16GB machine

Mục tiêu của bản rewrite này:
- Giữ pipeline gốc của tác giả: multimodal item embedding -> hierarchical graph -> joint recommendation + compatibility.
- Subsample khoảng 30k user trước, rồi chỉ giữ outfit và item thực sự liên quan.
- Chỉ embed ảnh có trong tập subsample, với tên file ảnh là `item_id.png`.
- Code gọn hơn, tách module rõ ràng, train xong sinh ra `model.pt` hoàn chỉnh để dùng cho app demo.

## Cấu trúc
- `hfgat/config.py`: config tập trung.
- `hfgat/data.py`: load data, subsample, build graph, dataset.
- `hfgat/features.py`: trích xuất image/text feature và tạo item init embedding.
- `hfgat/model.py`: H-HFGAT model.
- `hfgat/train.py`: train + evaluate + save `model.pt`.
- `hfgat/infer.py`: ví dụ load model và recommend.

## Cài đặt
```bash
pip install torch torchvision torch-geometric torch-scatter pandas numpy pillow transformers
```

## Chạy train
```bash
python -m hfgat.train \
  --data-root /path/to/authordata \
  --image-dir fashion_item_images \
  --work-dir /path/to/output_run \
  --user-target 30000 \
  --epochs 20 \
  --device cuda
```

Nếu máy yếu hơn, có thể chạy:
```bash
python -m hfgat.train \
  --data-root /path/to/authordata \
  --work-dir /path/to/output_run \
  --user-target 20000 \
  --epochs 10 \
  --device cpu
```

## Output
Sau khi train xong, thư mục `work-dir` sẽ có:
- `model.pt`: checkpoint đầy đủ, đã chứa `state_dict`, config, mappings, graph tensors, history.
- `training_history.json`
- `test_report.json`
- `sampled/item_data_sampled.csv`
- `sampled/outfit_data_sampled.csv`
- `sampled/user_data_sampled.csv`
- `cache/image_features.npy`
- `cache/text_features.npy`

## Inference demo
```bash
python -m hfgat.infer \
  --model-path /path/to/output_run/model.pt \
  --user-id 12345 \
  --topk 10
```

## Ghi chú kỹ thuật
- Mặc định dùng `resnet50` + `distilbert-base-uncased` để phù hợp máy 16GB RAM.
- Nếu muốn sát paper hơn, đổi `--image-model resnet152`.
- Nếu title không phải tiếng Anh, đổi text encoder sang model phù hợp hơn.
- Negative sampling hiện tại dùng random outfit cho recommendation và thay một item khác category cho compatibility.
