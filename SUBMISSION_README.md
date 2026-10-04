# DeepWeeds lab — hướng dẫn chạy và artifact nộp bài

Đây là README cho phần bài làm; README.md ở gốc repo là đề bài lab. Pipeline được chạy trên GPU local Windows, không phải notebook Colab/Kaggle. Vì vậy hiện chưa có URL notebook công khai.

## Môi trường đã ghi trong kết quả

- NVIDIA GeForce RTX 3050 Laptop GPU, CUDA.
- Python 3.13.5; PyTorch 2.12.0+cu130; torchvision 0.27.0+cu130.
- timm 1.0.15; NumPy 2.5.2; pandas 2.3.3; scikit-learn 1.9.1; matplotlib 3.11.2.
- huggingface_hub==0.34.4 được khóa theo requirements.txt để tương thích với timm; môi trường Python nằm trong .venv-k4lab và không commit.

## Dữ liệu và kết quả

Đặt ảnh ở data/images/ và CSV gốc ở data/labels/. images.zip và dữ liệu giải nén không commit. Summary đã ghi nhận fold 0 gồm train 10.501, val 3.501, test 3.507 ảnh; không giao nhau, đủ 17.509 ảnh.

Các lệnh chuẩn bị dữ liệu, huấn luyện, inference, score/grade và tạo workbook theo đúng thứ tự nằm trong [RUNBOOK.md](RUNBOOK.md). Config, checkpoint, history và validation predictions đã lưu dưới runs/, runs_final/, predictions/ và curves/. Không dùng test để chọn cấu hình.

## Tình trạng artifact

- Đã có 5 backbone, 13 cấu hình training/ablation, F01 seed 0 và phép đo latency one-view sơ bộ.
- Báo cáo thực nghiệm hiện tại: [report.md](report.md).
- **Chưa sẵn sàng nộp như kết quả chung kết:** chưa đủ 3 seed, chưa có test predictions, chưa có results.xlsx/điểm eval.py grade; final baseline T00 seed 0 chưa huấn luyện xong.
- Sau khi hoàn thành pipeline, đặt README này, report.md, results.xlsx, code, curves và prediction CSV vào submissions/<mssv>_<ho_ten_khong_dau>/. Nếu dùng notebook để nộp, bổ sung URL notebook tại đây.

Không commit dataset, images.zip, môi trường ảo hoặc checkpoint lớn.
