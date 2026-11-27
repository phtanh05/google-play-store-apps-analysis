# LLM Usage

Ghi nhận các lần sử dụng công cụ LLM đáng kể trong quá trình thực hiện đồ án (mục đích hỗ trợ học tập, rà soát và kiểm tra).

## Văn Đình Hiếu

### 1. Phân tích yêu cầu đề bài và định hướng câu hỏi
- **Công cụ:** Gemini LLM
- **Nội dung hỏi:** Hỏi để làm rõ các tiêu chí chiều sâu cho câu hỏi Non-ML (D2-D6), các yêu cầu bắt buộc của câu hỏi ML (chống rò rỉ dữ liệu, chia tập theo nhóm, baseline, error analysis) và cách diễn đạt câu hỏi nghiên cứu cho rõ ràng hơn.
- **Cách sử dụng / chỉnh sửa:** Dựa trên giải thích để tự kiểm tra lại ý tưởng cho câu 2.3 (độ mới của bản cập nhật) và câu 3.1 (dự đoán ứng dụng đạt mốc 1 triệu lượt cài), tự chọn biến và điều chỉnh lại câu chữ cho sát dữ liệu thực tế.

### 2. Rà soát code và xử lý lỗi ở câu 2.3 (`02_analysis.ipynb`)
- **Công cụ:** Gemini LLM
- **Nội dung hỏi:** Hỏi cách viết hàm tính chênh lệch theo tầng có trọng số và cách lấy mẫu bootstrap sao cho chạy nhanh, không bị lỗi chia cho 0 hoặc lỗi index khi lọc các tầng có ít mẫu.
- **Cách sử dụng / chỉnh sửa:** Tự viết lại hàm `q2_adj_diff` và `q2_estimate` theo cấu trúc vòng lặp đơn giản, tự kiểm tra lại logic lọc tầng (`min_n = 10`) và điều chỉnh các tham số kiểm tra độ bền (ngưỡng 30/90/180 ngày, phân tầng thêm `Size_group`).

### 3. Rà soát Pipeline và đánh giá mô hình ở câu 3.1 (`03_ml_question.ipynb`)
- **Công cụ:** Gemini LLM
- **Nội dung hỏi:** Hỏi cách gom bước tiền xử lý (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) vào chung `ColumnTransformer` và `Pipeline`, cách dùng `GroupShuffleSplit` / `GroupKFold` để không bị trùng tên app giữa các tập, và giải thích các lỗi cú pháp/cảnh báo khi chạy `GridSearchCV`.
- **Cách sử dụng / chỉnh sửa:** Tự triển khai các bước chia tập, chạy riêng từng mô hình (`DummyClassifier`, `LogisticRegression`, `HistGradientBoostingClassifier`) và điều chỉnh lại danh sách cột đầu vào (`CAT_COLS`, `NUM_COLS`, `TECH_COLS`) sau khi kiểm tra trùng lặp giữa `Type` và `Price`.

### 4. Kiểm tra và đối chiếu kết quả phân tích
- **Công cụ:** Gemini LLM
- **Nội dung hỏi:** Nhờ rà soát lại xem cách diễn giải các chỉ số (ROC-AUC, PR-AUC, F1, Permutation Importance, khoảng tin cậy bootstrap) và các con số tổng hợp ở phần trả lời có bị mâu thuẫn với bảng kết quả hay suy diễn quá mức (nhân quả) không.
- **Cách sử dụng / chỉnh sửa:** Tự đối chiếu lại với output thực tế trong notebook, chỉnh sửa các câu nhận xét cho thận trọng hơn (nhấn mạnh đây là mối liên hệ quan sát/dự báo) và bổ sung các giới hạn thực tế của dữ liệu.
