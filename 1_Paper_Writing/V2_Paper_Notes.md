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

Về đặc trưng đa phương thức cho item, mỗi item có ảnh và tiêu đề tiếng Trung. Chúng tôi sử dụng ResNet-152 cho ảnh và BERT Chinese cho văn bản, giống mô hình H-HFGAT. Vector visual và vector text được chiếu về cùng không gian 64 chiều, ghép nối, rồi qua một lớp tuyến tính để thu được embedding item ban đầu, gọi là *h_i*. Ở giai đoạn hiện tại lớp fusion này chưa được huấn luyện (giống cách H-HFGAT đã triển khai). Trọng số fusion được cố định sau khi khởi tạo ngẫu nhiên.

Về embedding user và outfit, user và outfit có embedding theo ID, khởi tạo ngẫu nhiên trong khoảng nhỏ quanh giá trị 0. Khác H-HFGAT (chỉ cố định tensor và huấn luyện phần GAT), Lightweight FGAT cho phép cập nhật embedding user, outfit và item trong quá trình tối ưu. Lý do là vector ID khởi tạo ngẫu nhiên cần thích nghi cùng lan truyền đồ thị. Nếu giữ embedding theo ID cố định và chỉ học các lớp GAT, mô hình khó bắt đủ sở thích riêng của từng user, nên chất lượng gợi ý thường thấp hơn so với cập nhật đồng thời.

Về tách giai đoạn trích xuất và huấn luyện đồ thị, ResNet và BERT chạy một lần (hoặc đọc từ cache), tạo vector cố định cho mỗi item. Các epoch sau chỉ cập nhật embedding và tham số GAT. Cách này tránh lặp lại chi phí trích xuất nặng mỗi lần thử hyperparameter. Pipeline cache và batch được mô tả thêm ở Mục 3.6.

---

### 3.3 Đồ thị item theo category và lan truyền attention

Tiểu mục này gồm hai phần: cách gán trọng số cạnh item-item (hybrid edge weighting), và cách attention cập nhật embedding item (per-destination softmax).

#### 3.3.1 Trọng số category và đồ thị item-item

H-HFGAT dùng thống kê đồng xuất hiện category trong toàn bộ outfit để đo mức “thường đi cùng nhau” giữa hai loại trang phục (ví dụ áo và quần). Với hai category c_i và c_j, trọng số thô được chuẩn hóa theo số lần xuất hiện của c_j:

w(c_i, c_j) = \frac{co(c_i, c_j)}{\sum_{c_k} co(c_i, c_k) / o(c_k)}

Trong đó co(c_i, c_j) là số outfit chứa cả hai category, o(c_j) là tổng số lần category c_j xuất hiện. Sau đó các giá trị được min-max chuẩn hóa trên toàn tập cặp category để đưa về khoảng thống nhất. Nếu hai category không có thống kê liên kết, chúng tôi gán trọng số mặc định nhỏ (0,1 trong triển khai).

Khi xây subgraph item cho từng outfit, mỗi cặp item (i, j) nhận trọng số cạnh bằng trọng số giữa category tương ứng. Item thuộc category hay kết hợp với nhau sẽ có cạnh mạnh hơn. Đây là nhánh có trọng số theo thống kê thời trang trong thiết kế hybrid.

Hai loại cạnh còn lại dùng trọng số 1: outfit-item (chỉ ghi membership), và user-outfit (chỉ ghi tương tác). Ba kiểu cạnh phục vụ ba vai trò khác nhau, đó là lý do gọi là hybrid edge weighting.


| Loại cạnh                    | Nguồn trọng số                      | Vai trò                                   |
| ---------------------------- | ----------------------------------- | ----------------------------------------- |
| item-item                    | Đồng xuất hiện category (chuẩn hóa) | Mã hóa quan hệ thời trang giữa các item   |
| outfit-item                  | 1,0                                 | Gom vector item thành outfit              |
| user-outfit (khi lan truyền) | 1,0, chỉ tập train                  | Gom vector outfit thành user, tránh rò rỉ |


#### 3.3.2 Attention đa đầu trên tầng item

Trên subgraph item của mỗi outfit, Lightweight FGAT áp dụng graph attention đa đầu (4 heads trong cấu hình hiện tại). Dưới đây chúng tôi ghi công thức cho một head; mỗi head có bộ tham số riêng, rồi ghép đầu ra trước dropout và chuẩn hóa L2.

Gọi h_i \in \mathbb{R}^d là embedding ban đầu của item i, và N_i là tập các item láng giềng của i trong subgraph (các item cùng outfit, nối với i qua cạnh item-item). Với ma trận học W, vector tham số attention \mathbf{a}, hệ số attention thô giữa i và láng giềng j \in N_i là

e_{ij} = \mathrm{LeakyReLU}\left(\mathbf{a}^{\top} \left[ W h_i  W h_j \right]\right),

trong đó  là phép nối hai vector.

So với H-HFGAT, chúng tôi chuẩn hóa softmax theo từng nút đích (per-destination): mỗi item i chỉ chuẩn hóa trên láng giềng của chính nó, không gom mọi cạnh trong subgraph vào một softmax chung. Trọng số attention là

\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N_i} \exp(e_{ik})}.

Trọng số cạnh category w_{ij} (Mục 3.3.1) được nhân vào thông điệp khi tổng hợp. Embedding item sau một lớp attention có dạng residual

h_i^{*} = h_i + \mathrm{LeakyReLU}\left(\sum_{j \in N_i} \alpha_{ij} W_1 \left(h_i \odot h_j\right)\right),

trong đó \odot là tích Hadamard (nhân từng phần tử), W_1 là ma trận học ở bước tổng hợp. Sau lớp item, mô hình áp dụng dropout và chuẩn hóa L2 trên đầu ra. Cách chuẩn hóa per-destination phù hợp với attention trên đồ thị: mỗi nút chỉ phân bổ trọng số trên hàng xóm của chính nó.

---

### 3.4 Tổng hợp outfit và user

Sau khi có h_i^{*}, mô hình lan truyền lên outfit rồi lên user theo hai bước attention tương tự H-HFGAT, nhưng có ràng buộc quan trọng về tập cạnh.

#### 3.4.1 Tầng outfit

Gọi N_o là tập các item thuộc outfit o, h_o là embedding ban đầu của outfit, và h_i^{*} là embedding item sau lớp attention (Mục 3.3.2). Với ma trận học W và vector tham số attention \mathbf{a}, hệ số attention thô giữa item i \in N_o và outfit o là

e_{io} = \mathrm{LeakyReLU}\left(\mathbf{a}^{\top} \left[ W h_i^{*}  W h_o \right]\right).

Trọng số attention được chuẩn hóa per-destination trên các item trong cùng outfit:

\alpha_{io} = \frac{\exp(e_{io})}{\sum_{j \in N_o} \exp(e_{jo})}.

Embedding outfit sau bước tổng hợp có dạng residual

h_o^{*} = h_o + \mathrm{LeakyReLU}\left(\sum_{i \in N_o} \alpha_{io} W_2 h_i^{*}\right).

Vector h_o^{*} gom tín hiệu compatibility ở mức item: item nào được attention cao hơn sẽ đóng góp mạnh hơn vào biểu diễn outfit.

#### 3.4.2 Tầng user

Gọi N_u là tập các outfit mà user u đã tương tác, h_u là embedding ban đầu của user, và h_o^{*} là embedding outfit sau Mục 3.4.1. Hệ số attention thô giữa outfit o \in N_u và user u là

e_{ou} = \mathrm{LeakyReLU}\left(\mathbf{a}^{\top} \left[ W h_o^{*}  W h_u \right]\right).

Trọng số attention được chuẩn hóa per-destination trên các outfit của cùng user:

\alpha_{ou} = \frac{\exp(e_{ou})}{\sum_{j \in N_u} \exp(e_{ju})}.

Embedding user sau bước tổng hợp có dạng residual

h_u^{*} = h_u + \mathrm{LeakyReLU}\left(\sum_{o \in N_u} \alpha_{ou} W_3 h_o^{*}\right).

Về ràng buộc train-only, khi huấn luyện và khi chạy forward trên validation, N_u chỉ gồm outfit thuộc tập train của user đó. Tương tác validation và test không tham gia lan truyền lên h_u^{*} và h_o^{*}. Đồ thị đầy đủ có thể lưu mọi cạnh user-outfit để tiện quản lý dữ liệu, nhưng message passing bỏ qua cạnh ngoài train. Cách này ngăn mô hình “nhìn thấy” sở thích hold-out trong khi cập nhật biểu diễn, khác với thiết lập H-HFGAT dùng toàn bộ cạnh khi train.

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

Về tách nhánh compatibility (decoupled compatibility), điểm compatibility tính trên **embedding item gốc** h_i (trước GAT), không dùng h_i^{*}. Gradient từ loss compatibility không cập nhật embedding item gốc và không lan ngược qua đường lan truyền item GAT; chỉ **CompatibilityScorer** được học qua nhánh này. Cách tách này tránh hai loss cùng kéo geometry item theo hai hướng đối nghịch.

Huấn luyện compatibility theo Fill In The Blank: mỗi outfit đúng đi với ba outfit sai, mỗi outfit sai thay một item bằng item khác outfit. Outfit đúng phải có điểm cao hơn outfit sai.

#### 3.5.3 Hàm mất mát BPR và kết hợp đa nhiệm vụ

Cả recommendation và compatibility đều dùng **Bayesian Personalized Ranking (BPR)**. Với cặp dương-âm (u, o, o') cho gợi ý:

\mathcal{L}*{\mathrm{rec}} = -\sum \log \sigma\left(y*{uo} - y_{uo'}\right)

Với cặp outfit dương-âm (o, o') cho compatibility:

\mathcal{L}*{\mathrm{comp}} = -\sum \log \sigma\left(s_o - s*{o'}\right)

Hàm mục tiêu chung:

\mathcal{L} = \mathcal{L}*{\mathrm{rec}} + \lambda \mathcal{L}*{\mathrm{comp}}

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

Ngoài phần công thức, chúng tôi sắp xếp Lightweight FGAT theo pipeline tuần tự từ dữ liệu thô đến đánh giá hold-out, để không phải lặp lại các bước tốn kém mỗi lần chỉnh hyperparameter.

Chúng tôi bắt đầu bằng các dữ liệu user-outfit thô, lọc sub-graph trước khi đưa vào mô hình. User phải có ít nhất bốn tương tác để chia train, validation và test theo từng người, đủ dày cho lan truyền ở tầng user, đồng thời tạo bộ Fill In The Blank cho compatibility. Lọc cascade bỏ outfit và item không đủ điều kiện. Các item bị thiếu ảnh cũng bị loại để giảm dữ liệu nhiễu. So với corpus gốc, đồ thị sau bước này nhỏ hơn đáng kể.

ResNet-152 và BERT Chinese trích xuất đặc trưng ảnh và tiêu đề item (batch 64 ảnh, batch 128 văn bản). Visual vectors và text vectors được lưu cache sau lần chạy đầu. Các lần sau bỏ qua hai bước xử lý này nếu dữ liệu cache còn dùng được. Đây chỉ là tối ưu thực thi, không đổi cách định nghĩa đặc trưng ở Mục 3.2.

Từ cache và seed, chúng tôi khởi tạo embedding item, outfit và user. Lớp fusion tuyến tính cho item vẫn chưa huấn luyện, như ở Mục 3.2. Tiếp theo, chúng tôi dựng ba ma trận thưa item-item, outfit-item và user-outfit. Trọng số category (Mục 3.3.1) gán cho cạnh item-item. Ma trận user-outfit giữ đủ các tương tác, riêng bước lan truyền lên user chỉ dùng outfit thuộc tập train.

Huấn luyện chạy tối đa 50 epoch với batch size 512. Mỗi outfit dương trong BPR gợi ý có ba outfit âm. Dropout 0,3 ở tầng item, weight decay 10^{-5}, learning rate 0,001. ReduceLROnPlateau giảm learning rate khi Hit Rate@10 trên validation không tăng. Checkpoint chọn theo Hit Rate@10, early stopping với patience 10 epoch. Các lớp GAT và CompatibilityScorer cập nhật theo hàm mục tiêu ở Mục 3.5.

Sau huấn luyện, chúng tôi đo hiệu năng trên validation và test: Metric xếp hạng cho gợi ý và độ chính xác Fill In The Blank cho compatibility. Với gợi ý, mỗi outfit dương được xếp hạng cùng 50 outfit âm ngẫu nhiên, đã bỏ các outfit user đã có trong train.

---

## Ghi chú biên tập

- **Hình 3:** Chèn tham chiếu “(xem Hình 3)” khi dán vào PDF V2.
- **Công thức:** Khi chuyển sang Word/LaTeX, giữ đánh số phương trình liên tục (1), (2), … theo LNCS.
- **Thuật ngữ giữ tiếng Anh:** graph attention, embedding, Fill In The Blank, BPR, Hit Rate@10, softmax, dropout, ResNet-152, BERT, compatibility, recommendation.
- **Bước tiếp theo (do tác giả kích hoạt):** `/write-lncs-paper` chế độ VI→EN cho toàn Mục 3, rồi humanize bản tiếng Anh trước khi nộp LNCS.

