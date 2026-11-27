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

### 5. Rà soát bố cục và trình bày tài liệu (`README.md`)

* **Công cụ:** Gemini LLM
* **Nội dung hỏi:** Đưa bản nháp `README.md` tự viết để nhờ gợi ý cách sắp xếp lại bố cục các đề mục, cải thiện định dạng Markdown (bảng biểu, cây thư mục, khối lệnh) giúp người đọc dễ theo dõi cấu trúc đồ án và cách chạy mã nguồn.
* **Cách sử dụng / chỉnh sửa:** Giữ nguyên nội dung cốt lõi tự soạn, áp dụng các gợi ý về phân cấp tiêu đề và định dạng trình bày, đồng thời tự rà soát lại toàn bộ tên file, đường dẫn thư mục và câu lệnh cài đặt để đảm bảo khớp hoàn toàn với thực tế của repository.

## Nguyễn Trung Dũng
### 1. Kiểm tra tính nhất quán của dữ liệu đầu vào và kết quả tính toán ở câu 2.5 và 2.6 (`02_analysis.ipynb`)

- **Công cụ:** Gemini LLM (Hydrogen)
- **Nội dung hỏi:** Nhờ đối chiếu lại các bước lọc dữ liệu (xử lý giá trị thiếu, loại trùng lặp theo đơn vị phân tích) và kiểm tra xem các con số thống kê tổng hợp (tỷ lệ phần trăm, chênh lệch điểm phần trăm, giá trị trung vị/trung bình) ở câu 2.5 và 2.6 có bị mâu thuẫn với bảng kết quả xuất ra trong notebook hay không.
- **Cách sử dụng / chỉnh sửa:** Tự thực hiện toàn bộ phần lập trình, thiết kế biến và chạy ra kết quả gốc trước; sau đó dùng phản hồi của AI để rà soát lại các phép tính làm tròn số, kiểm tra chéo giữa biểu đồ và bảng số liệu nhằm đảm bảo độ chính xác tuyệt đối.

### 2. Rà soát cách diễn giải kết quả và trau chuốt nhận xét phân tích (`01_collection_exploration.ipynb`,`02_analysis.ipynb`)

- **Công cụ:** Gemini LLM (Hydrogen)
- **Nội dung hỏi:** Đưa các đoạn nhận xét tự viết cho câu 2.5 và 2.6 để nhờ kiểm tra xem cách diễn đạt có bị suy diễn quá mức (nhầm lẫn giữa tương quan quan sát được với quan hệ nhân quả) hay bỏ sót ý quan trọng nào từ biểu đồ/bảng biểu không.
- **Cách sử dụng / chỉnh sửa:** Giữ nguyên các luận điểm chính tự rút ra từ dữ liệu thực tế; điều chỉnh lại câu chữ theo hướng khách quan, thận trọng và mạch lạc hơn, đồng thời bổ sung rõ ràng phạm vi diễn giải và giới hạn của tập dữ liệu gốc.

### 3. Chuẩn hóa văn phong học thuật và bố cục trình bày trong Notebook(`01_collection_exploration.ipynb`,`02_analysis.ipynb`)

- **Công cụ:** Gemini LLM (Hydrogen)
- **Nội dung hỏi:** Nhờ gợi ý cách phân tách các ý nhận xét thành gạch đầu dòng ngắn gọn và thống nhất cách dùng thuật ngữ xuyên suốt từ phần đặt câu hỏi, động lực, tiêu chí chiều sâu đến phần kết luận ở cả hai câu 2.5 và 2.6.
- **Cách sử dụng / chỉnh sửa:** Áp dụng các gợi ý về cách ngắt đoạn, in đậm từ khóa trọng tâm và tự rà soát lại toàn bộ văn bản trong các ô Markdown của `02_analysis.ipynb` để đảm bảo đồng bộ về hình thức với các phần còn lại của nhóm.