# V2 Paper Notes: Bản nháp tiếng Việt

> **Phạm vi file này:** Mục 3 Phương pháp đề xuất (Proposed Method) cho báo cáo [V2].  
> **Trạng thái:** Đã humanize (tiếng Việt). Chưa dịch sang tiếng Anh LNCS.  
> **Nhánh tham chiếu:** `kimnguyen-quoctran-v1.3`  
> **Tên phương pháp:** **Lightweight FGAT** (của chúng tôi). Baseline tác giả: **H-HFGAT** (định nghĩa một lần ở đầu Mục 3, sau đó dùng tên này khi so sánh).

---

## 3. Phương pháp đề xuất

Gợi ý trang phục trong thương mại điện tử cần xử lý hai việc cùng lúc. Hệ thống phải đo mức độ hợp nhau giữa các món trong một outfit, và phải xếp hạng outfit phù hợp với từng người dùng. **H-HFGAT** (Hybrid-Hierarchical Fashion Graph Attention Network) giải quyết cả hai trong một đồ thị ba tầng user, outfit và item, với lan truyền từ item lên outfit rồi lên user.

Chúng tôi xây dựng **Lightweight FGAT** dựa trên khung đó. Điểm khác biệt nằm ở cách khởi tạo và cập nhật embedding, cách gán trọng số cạnh, cách chuẩn hóa attention, cách tách hai nhiệm vụ khi huấn luyện, và cách tổ chức pipeline dữ liệu để chạy ổn định trên tài nguyên hạn chế. Phần còn lại của mục này mô tả từng khối theo thứ tự xử lý thực tế của hệ thống.

---



### 3.1 Tổng quan

Về cấu trúc đồ thị, mô hình tổ chức ba lớp nút user, outfit và item. Ba nhóm quan hệ chính là user-outfit (lịch sử tương tác), outfit-item (thành phần của outfit) và item-item (các item xuất hiện chung trong cùng outfit, có trọng số theo mức đồng xuất hiện category).

Về luồng xử lý tổng thể, dữ liệu thô gồm tương tác user-outfit, thành phần từng outfit, ảnh và tiêu đề item. Pipeline lọc người dùng và entity không đủ điều kiện, trích xuất đặc trưng ảnh và văn bản, khởi tạo ma trận embedding, xây đồ thị thưa, chia tập train/validation/test, rồi huấn luyện mô hình đồ thị. Ở đầu ra có hai nhánh: (i) điểm gợi ý giữa user và outfit đã cập nhật, (ii) điểm compatibility trên outfit dùng embedding item gốc.

Về hai nhiệm vụ học, nhiệm vụ gợi ý (recommendation) học vector user và outfit sau lan truyền để xếp hạng outfit cho từng user. Nhiệm vụ compatibility học đánh giá outfit có nhất quán hay không, qua bài toán Fill In The Blank: một outfit đúng đối chiếu với các outfit sai tạo bằng cách thay một item.

Về lý do cần phiên bản Lightweight, H-HFGAT dùng cùng backbone ResNet-152 và BERT, nhưng pipeline tái tạo nặng, embedding ID cố định, lan truyền user-outfit có thể chứa tương tác validation, và compatibility gắn chặt với đường lan truyền GAT. Lightweight FGAT giữ backbone đặc trưng, đổi giao thức huấn luyện và đánh giá để giảm rò rỉ dữ liệu, ổn định đa nhiệm vụ, và rút ngắn thời gian lặp lại thí nghiệm. Chi tiết từng thay đổi nằm ở các tiểu mục 3.2-3.6.

Về hình minh họa, cấu trúc tổng thể được tóm tắt ở Hình 3. Từ trái sang phải: đặc trưng đa phương thức, embedding khởi tạo, ba lớp lan truyền (item, outfit, user), và hai đầu chấm điểm (recommendation và compatibility).

---



### 3.2 Khởi tạo embedding item

Về đặc trưng đa phương thức cho item, mỗi item có ảnh và tiêu đề tiếng Trung. Chúng tôi dùng ResNet-152 cho ảnh và BERT Chinese cho văn bản, cùng họ mô hình với H-HFGAT. Vector visual và vector text được chiếu về cùng không gian 64 chiều, ghép nối, rồi qua một lớp tuyến tính để thu được embedding item ban đầu h_i. Ở giai đoạn hiện tại lớp fusion này **chưa được huấn luyện** (giống triển khai tác giả). Trọng số fusion cố định sau khởi tạo ngẫu nhiên.

Về embedding user và outfit, user và outfit có embedding theo ID, khởi tạo ngẫu nhiên trong khoảng nhỏ quanh 0. Khác H-HFGAT (chỉ cố định tensor và huấn luyện phần GAT), Lightweight FGAT **cho phép cập nhật** embedding user, outfit và item trong quá trình tối ưu. Lý do là vector ID khởi tạo ngẫu nhiên cần thích nghi cùng lan truyền đồ thị. Chỉ huấn luyện lớp GAT mà giữ embedding đóng băng thường kém hơn trên split theo từng user.

Về tách giai đoạn trích xuất và huấn luyện đồ thị, ResNet và BERT chạy một lần (hoặc đọc từ cache), tạo vector cố định cho mỗi item. Các epoch sau chỉ cập nhật embedding và tham số GAT. Cách này tránh lặp lại chi phí trích xuất nặng mỗi lần thử hyperparameter. Pipeline cache và batch được mô tả thêm ở Mục 3.6.

Về giới hạn đã biết (không thuộc phạm vi huấn luyện hiện tại), huấn luyện lớp fusion visual-text là hướng cải tiến dự kiến, chưa triển khai trong phiên bản đang báo cáo.

---



### 3.3 Đồ thị item theo category và lan truyền attention

Tiểu mục này gồm hai phần: cách gán trọng số cạnh item-item (hybrid edge weighting), và cách attention cập nhật embedding item (per-destination softmax).

#### 3.3.1 Trọng số category và đồ thị item-item

H-HFGAT dùng thống kê đồng xuất hiện category trong toàn bộ outfit để đo mức “thường đi cùng nhau” giữa hai loại trang phục (ví dụ áo và quần). Với hai category c_i và c_j, trọng số thô được chuẩn hóa theo số lần xuất hiện của c_j:


w(c_i, c_j) = \frac{co(c_i, c_j)}{\sum_{c_k} co(c_i, c_k) / o(c_k)}


Trong đó co(c_i, c_j) là số outfit chứa cả hai category, o(c_j) là tổng số lần category c_j xuất hiện. Sau đó các giá trị được **min-max chuẩn hóa** trên toàn tập cặp category để đưa về khoảng thống nhất. Nếu hai category không có thống kê liên kết, chúng tôi gán trọng số mặc định nhỏ (0,1 trong triển khai).

Khi xây subgraph item cho từng outfit, mỗi cặp item (i, j) nhận trọng số cạnh bằng trọng số giữa category tương ứng. Item thuộc category hay kết hợp với nhau sẽ có cạnh mạnh hơn. Đây là nhánh **có trọng số theo thống kê thời trang** trong thiết kế hybrid.

Hai loại cạnh còn lại dùng trọng số 1: outfit-item (chỉ ghi membership), và user-outfit (chỉ ghi tương tác). Ba kiểu cạnh phục vụ ba vai trò khác nhau, đó là lý do gọi là **hybrid edge weighting**.


| Loại cạnh                    | Nguồn trọng số                      | Vai trò                                   |
| ---------------------------- | ----------------------------------- | ----------------------------------------- |
| item-item                    | Đồng xuất hiện category (chuẩn hóa) | Mã hóa quan hệ thời trang giữa item       |
| outfit-item                  | 1,0                                 | Gom vector item thành outfit              |
| user-outfit (khi lan truyền) | 1,0, chỉ tập train                  | Gom vector outfit thành user, tránh rò rỉ |




#### 3.3.2 Attention đa đầu trên tầng item

Trên mỗi subgraph item, Lightweight FGAT áp dụng graph attention đa đầu (4 heads trong cấu hình hiện tại). Hệ số attention thô giữa item i và láng giềng j:


e_{ij} = \mathrm{LeakyReLU}\left(\mathbf{a}^T [W h_i  W h_j]\right)


So với H-HFGAT, chúng tôi chuẩn hóa softmax theo **từng nút đích** (per-destination), không gom softmax toàn cục trên mọi cạnh. Mỗi item i có phân phối trọng số riêng trên tập láng giềng N_i:


\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N_i} \exp(e_{ik})}


Trọng số cạnh category w_{ij} được đưa vào quá trình tổng hợp thông điệp. Bản cập nhật residual:


h_i^{*} = h_i + \mathrm{LeakyReLU}\left(\sum_{j \in N_i} \alpha_{ij} W_1 (h_i \odot h_j)\right)


Sau lớp item áp dụng dropout và chuẩn hóa L2 trên đầu ra. Per-destination softmax phù hợp với định nghĩa attention trên đồ thị: mỗi nút chỉ phân bổ “sự chú ý” trên hàng xóm của chính nó.

---



### 3.4 Tổng hợp outfit và user

Sau khi có h_i^{*}, mô hình lan truyền lên outfit rồi lên user theo hai bước attention tương tự H-HFGAT, nhưng có ràng buộc quan trọng về tập cạnh.

#### 3.4.1 Tầng outfit

Gọi N_o là tập item thuộc outfit o. Attention từ item đã cập nhật h_i^{*} và embedding outfit ban đầu h_o:


e_{io} = \mathrm{LeakyReLU}\left(\mathbf{a}^T [W h_i^{*}  W h_o]\right), \quad
\alpha_{io} = \frac{\exp(e_{io})}{\sum_{j \in N_o} \exp(e_{jo})}



h_o^{*} = h_o + \mathrm{LeakyReLU}\left(\sum_{i \in N_o} \alpha_{io} W_2 h_i^{*}\right)


Vector h_o^{*} tích hợp tín hiệu compatibility ở mức item (item nào đóng góp mạnh vào outfit).

#### 3.4.2 Tầng user

Gọi N_u là tập outfit user u đã tương tác. Attention và cập nhật:


e_{ou} = \mathrm{LeakyReLU}\left(\mathbf{a}^T [W h_o^{*}  W h_u]\right), \quad
\alpha_{ou} = \frac{\exp(e_{ou})}{\sum_{j \in N_u} \exp(e_{ju})}



h_u^{*} = h_u + \mathrm{LeakyReLU}\left(\sum_{o \in N_u} \alpha_{ou} W_3 h_o^{*}\right)


Về ràng buộc train-only, khi huấn luyện và khi chạy forward trên validation, N_u **chỉ gồm outfit thuộc tập train** của user đó. Tương tác validation và test không tham gia lan truyền lên h_u^{*} và h_o^{*}. Đồ thị đầy đủ có thể lưu mọi cạnh user-outfit để tiện quản lý dữ liệu, nhưng message passing bỏ qua cạnh ngoài train. Cách này ngăn mô hình “nhìn thấy” sở thích hold-out trong khi cập nhật biểu diễn, khác với thiết lập H-HFGAT dùng toàn bộ cạnh khi train.

---



### 3.5 Mục tiêu huấn luyện và thay đổi giao thức

Lightweight FGAT huấn luyện đồng thời gợi ý và compatibility. Tiểu mục này nêu cách chấm điểm, hàm mất mát, và các thay đổi giao thức so với H-HFGAT. Chi tiết chia tập và metric đánh giá nằm ở Mục 4 (Thực nghiệm).

#### 3.5.1 Điểm gợi ý (recommendation)

H-HFGAT dùng tích vô hướng thô giữa h_u^{*} và h_o^{*}. Lightweight FGAT **chuẩn hóa L2** hai vector trước khi tính tích vô hướng:


y_{uo} = \hat{h}_u^{*\top} \hat{h}_o^{*}, \quad
\hat{h} = \frac{h}{h_2}


Điểm số mang tính cosine-style, giúp BPR ổn định hơn khi norm vector khác nhau giữa user.

#### 3.5.2 Điểm compatibility

Compatibility đánh giá outfit có hợp lý về mặt kết hợp item hay không. Thay vì mô tả đầy đủ ma trận attention A và compatibility C như H-HFGAT, Lightweight FGAT dùng **CompatibilityScorer** gọn hơn trên tensor item của outfit.

Về tách nhánh compatibility (decoupled compatibility), điểm compatibility tính trên **embedding item gốc** h_i (trước GAT), không dùng h_i^{*}. Gradient từ loss compatibility **không kéo** embedding gốc và không kéo đường lan truyền item GAT khi chế độ tách nhánh bật. CompatibilityScorer vẫn học trọng số riêng. Mục tiêu là tránh hai loss cùng kéo geometry item theo hai hướng đối nghịch.

Huấn luyện compatibility theo Fill In The Blank: mỗi outfit đúng đi với ba outfit sai, mỗi outfit sai thay một item bằng item khác outfit. Outfit đúng phải có điểm cao hơn outfit sai.

#### 3.5.3 Hàm mất mát BPR và kết hợp đa nhiệm vụ

Cả recommendation và compatibility đều dùng **Bayesian Personalized Ranking (BPR)**. Với cặp dương-âm (u, o, o') cho gợi ý:


\mathcal{L}*{rec} = -\sum \log \sigma(y*{uo} - y_{uo'})


Với cặp outfit dương-âm (o, o') cho compatibility:


\mathcal{L}*{comp} = -\sum \log \sigma(s_o - s*{o'})


Hàm mục tiêu chung:


\mathcal{L} = \mathcal{L}*{rec} + \lambda \mathcal{L}*{comp}


Trong thí nghiệm báo cáo, \lambda = 0{,}3. Trọng số này cân bằng xếp hạng và compatibility. Checkpoint tốt nhất chọn theo **Hit Rate@10** trên validation, không chọn theo tổng loss validation.

#### 3.5.4 Tóm tắt thay đổi giao thức so với H-HFGAT


| Khía cạnh                       | H-HFGAT (ý chính)                     | Lightweight FGAT                                   |
| ------------------------------- | ------------------------------------- | -------------------------------------------------- |
| Chia dữ liệu gợi ý              | Chia theo dòng file (90/10 train-val) | Chia 80/10/10 **theo từng user**                   |
| Cạnh user-outfit khi lan truyền | Có thể gồm validation                 | **Chỉ train**                                      |
| Embedding ID                    | Cố định                               | **Có thể học**                                     |
| Chuẩn hóa attention item        | Softmax toàn cục                      | **Per-destination**                                |
| Điểm gợi ý                      | Tích vô hướng thô                     | **Cosine-style** (L2 rồi tích vô hướng)            |
| Input compatibility             | Vector item sau GAT                   | **Embedding item gốc**, tách gradient              |
| Negative khi đánh giá gợi ý     | Random, có thể trùng outfit train     | Random, **loại outfit đã có trong train** của user |


Các thay đổi trên nhằm đánh giá gợi ý nghiêm hơn và huấn luyện đa nhiệm vụ ổn định hơn, không thay đổi backbone ResNet/BERT.

---



### 3.6 Pipeline hiệu quả

Lightweight FGAT không chỉ đổi công thức mô hình mà còn tổ chức pipeline để giảm chi phí lặp thí nghiệm. Bảng 1 tóm tắt các bước chính.

**Bảng 1.** Các bước pipeline Lightweight FGAT


| Bước                    | Đầu vào                      | Đầu ra                                 | Ghi chú                                                                         |
| ----------------------- | ---------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- |
| 1. Lọc dữ liệu          | Bản ghi user-outfit thô      | Subgraph đã lọc                        | User có ít nhất 4 tương tác; lọc cascade outfit/item; bỏ item thiếu ảnh nếu cần |
| 2. Trích xuất đặc trưng | Ảnh và tiêu đề item          | Vector visual và text                  | ResNet-152 + BERT Chinese, xử lý theo batch                                     |
| 3. Cache đặc trưng      | Vector vừa trích xuất        | File cache tái sử dụng                 | Lần chạy sau bỏ qua ResNet/BERT nếu cache còn hợp lệ                            |
| 4. Khởi tạo embedding   | Đặc trưng + seed             | Ma trận item, outfit, user             | Fusion tuyến tính chưa huấn luyện cho item                                      |
| 5. Xây đồ thị           | Subgraph + trọng số category | Ma trận thưa II, OI, UO                | Cạnh UO đầy đủ lưu trữ, lan truyền dùng tập train                               |
| 6. Chia tập             | Tương tác và outfit          | Train / val / test + Fill In The Blank | Split per-user cho gợi ý                                                        |
| 7. Huấn luyện           | Đồ thị, embedding, split     | Mô hình GAT + CompatibilityScorer      | Early stopping theo HR@10, tối đa 50 epoch                                      |
| 8. Đánh giá             | Tập hold-out                 | Metric xếp hạng và compatibility       | 50 negative/outfit khi đánh giá gợi ý                                           |


Về lọc người dùng, ngưỡng tối thiểu bốn tương tác giúp mỗi user có đủ lịch sử để chia train/validation/test cục bộ và để lan truyền user có nghĩa. Đồng thời giảm quy mô đồ thị so với corpus đầy đủ.

Về batch và cache, ảnh xử lý theo lô 64, văn bản theo lô 128 trong cấu hình hiện tại. Cache lưu vector đặc trưng sau lần trích xuất đầu. Đây là tối ưu kỹ thuật, không đổi định nghĩa backbone.

Về huấn luyện, batch size 512, mỗi outfit dương có ba outfit âm trong BPR gợi ý, dropout 0,3 trên lớp item, weight decay 10^{-5}, learning rate 0,001 với ReduceLROnPlateau khi HR@10 không cải thiện. Patience early stopping bằng 10 epoch.

Về hướng cải tiến pipeline (chưa triển khai), cắt tỉa láng giềng item-item theo top-K và huấn luyện lớp fusion vẫn là đề xuất tương lai, không nằm trong phạm vi mô tả chi tiết ở đây.

---



## Ghi chú biên tập

- **Hình 3:** Chèn tham chiếu “(xem Hình 3)” khi dán vào PDF V2.
- **Công thức:** Khi chuyển sang Word/LaTeX, giữ đánh số phương trình liên tục (1), (2), … theo LNCS.
- **Thuật ngữ giữ tiếng Anh:** graph attention, embedding, Fill In The Blank, BPR, Hit Rate@10, softmax, dropout, ResNet-152, BERT, compatibility, recommendation.
- **Bước tiếp theo (do tác giả kích hoạt):** `/write-lncs-paper` chế độ VI→EN cho toàn Mục 3, rồi humanize bản tiếng Anh trước khi nộp LNCS.

