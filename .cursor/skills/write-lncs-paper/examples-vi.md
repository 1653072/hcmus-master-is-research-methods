# Ví dụ viết LNCS: Tiếng Việt (bản nháp)

Mẫu câu cho các mục thường gặp. Điều chỉnh theo nội dung thực tế; không copy nguyên văn.

---

## Tóm tắt (~120 từ, bản nháp: bản nộp phải là tiếng Anh)

> Hệ thống gợi ý trang phục cần mô hình hóa cả sở thích người dùng và độ tương thích giữa các món trong outfit. Các phương pháp graph attention hiện có thường huấn luyện trên đồ thị có rò rỉ tương tác tương lai và đánh giá compatibility chưa thống nhất. Chúng tôi trình bày pipeline graph attention nhẹ (Lightweight) chỉ dùng cạnh user–outfit từ tập huấn luyện, tách chấm điểm compatibility khỏi xếp hạng gợi ý, và báo cáo metric riêng cho từng nhiệm vụ. Trên Polyvore, mô hình đạt Hit Rate@10 validation 0,77 và NDCG@10 0,46, với độ chính xác Fill In The Blank 0,72. Thiết kế ưu tiên ổn định xếp hạng trong kịch bản active-user. Mã nguồn sẽ công bố khi được chấp nhận.

**Từ khóa (khi chuyển sang EN):** Outfit recommendation · Graph attention · Compatibility · Polyvore

---

## Giới thiệu: đoạn mở

> Gợi ý outfit giúp người dùng khám phá tổ hợp trang phục phù hợp phong cách. Các hướng gần đây biểu diễn user, item và outfit dưới dạng đồ thị và dùng attention để lan truyền tín hiệu quan hệ. Tuy vậy, cách xây đồ thị và giao thức đánh giá ảnh hưởng mạnh đến kết quả báo cáo, và metric compatibility giữa các bài thường khó so sánh trực tiếp.

## Giới thiệu: đóng góp (đánh số)

> Bài báo có các đóng góp sau:
> 1. Chúng tôi mô tả pipeline graph attention nhẹ với đồ thị user–outfit chỉ từ dữ liệu huấn luyện, tránh rò rỉ ở thời điểm đánh giá.
> 2. Chúng tôi tách compatibility scoring khỏi recommendation ranking và ghi rõ giao thức đánh giá cho từng nhiệm vụ.
> 3. Chúng tôi báo cáo kết quả trên Polyvore và phân tích khác biệt giao thức so với baseline FGAT đã công bố.

## Giới thiệu: dẫn cấu trúc

> Cấu trúc bài báo như sau: Mục 2 tóm tắt nghiên cứu liên quan, Mục 3 trình bày phương pháp, Mục 4 mô tả thực nghiệm, Mục 5 kết luận.

---

## Nghiên cứu liên quan: đoạn theo chủ đề

> **Gợi ý outfit dựa trên đồ thị.** Các công trình sớm mô hình outfit như tập item và học compatibility bằng metric learning hoặc decoder tự hồi quy. Graph attention mở rộng hướng này bằng lan truyền trên cạnh item–item và user–outfit. FGAT tối ưu đồng thời gợi ý và Fill In The Blank với encoder chung. Khác FGAT, chúng tôi chỉ dùng tương tác huấn luyện khi xây đồ thị user–outfit và báo cáo accuracy compatibility tách khỏi metric xếp hạng.

---

## Phương pháp: mở đầu tiểu mục

> **Bài toán.** Gọi $\mathcal{U}$, $\mathcal{I}$, $\mathcal{O}$ là tập user, item và outfit. Mỗi outfit là tập con của item. Nhiệm vụ gợi ý xếp hạng outfit cho user; nhiệm vụ compatibility đánh giá outfit có nhất quán nội bộ hay không.

> **Chấm compatibility.** Từ embedding item $\mathbf{e}_i$, chúng tôi chấm outfit bằng tổng hợp bất biến hoán vị trên các item, huấn luyện với negative Fill In The Blank.

---

## Thực nghiệm: thiết lập và lưu ý

> **Dữ liệu và chia tập.** Chúng tôi dùng split Polyvore với train, validation, test cho gợi ý và tập Fill In The Blank riêng cho compatibility. Cạnh user–outfit trong lan truyền đồ thị chỉ lấy từ train.

> **Metric.** Chúng tôi báo cáo Hit Rate@10 và NDCG@10 cho gợi ý, accuracy cho Fill In The Blank. Precision@10 nhạy với số outfit ground-truth trên validation nên chúng tôi nhấn mạnh Hit Rate@10 và NDCG@10 khi so sánh.

## Thực nghiệm: diễn giải kết quả

> Bảng 1 cho thấy phương pháp của chúng tôi cải thiện Hit Rate@10 và NDCG@10 so với FGAT trên test trong bài gốc, trong khi accuracy compatibility thấp hơn. Khoảng cách compatibility một phần do negative sampling khác và thiết kế tách nhánh compatibility để ổn định xếp hạng. Do đó chúng tôi coi số liệu compatibility mang tính tham chiếu, không phải so sánh head-to-head hoàn toàn công bằng.

---

## Chú thích bảng (phía trên bảng)

> **Bảng 1.** Kết quả gợi ý và compatibility. FGAT lấy từ bài gốc (test). Kết quả của chúng tôi trên validation tại checkpoint xếp hạng tốt nhất. Accuracy compatibility theo Fill In The Blank.

| Metric | FGAT (bài gốc, test) | Lightweight (val., tốt nhất) |
|--------|---------------------:|-----------------------------:|
| HR@10 | 0,4286 | 0,7737 |
| NDCG@10 | 0,1340 | 0,4645 |
| Accuracy compatibility | 0,8956 | 0,7195 |

*Lưu ý: bản nộp LNCS dùng dấu chấm thập phân (0.7737) trong bảng tiếng Anh.*

---

## Chú thích hình (phía dưới hình)

> **Hình 1.** Tổng quan pipeline graph attention nhẹ. Embedding item và user được cập nhật qua ba lớp đồ thị; compatibility và recommendation dùng đầu chấm điểm riêng.

---

## Kết luận

> Chúng tôi trình bày pipeline graph attention nhẹ cho gợi ý outfit với đồ thị chỉ từ train và đánh giá compatibility tách biệt. Thực nghiệm trên Polyvore cho thấy metric xếp hạng tốt trên validation, còn accuracy compatibility còn dư địa cải thiện. Hướng phát triển gồm huấn luyện lớp fusion visual–text và cắt tỉa láng giềng item–item để lan truyền thưa hơn.

---

## Cụm từ nên tránh → nên dùng

| Tránh | Nên dùng |
|-------|----------|
| Ở phần này chúng tôi sẽ trình bày… | Mục 4 đánh giá… |
| Điều đáng lưu ý là… | (nêu thẳng sự kiện) |
| Hiệu năng state-of-the-art | Hit Rate@10 đạt X trên Y |
| Người đọc cần lưu ý… | Chúng tôi lưu ý… / (bỏ) |
| Rõ ràng là / Hiển nhiên | (bỏ hoặc chứng minh) |
| `compat_acc` trong văn bản | độ chính xác compatibility |
| Em dash (`—`) | Dấu phẩy, chấm, hoặc ngoặc đơn |
| Active-User | Lightweight (tên phương pháp trong báo cáo) |
| Trộn 71,95% và 0,7195 | Một định dạng trong cùng một bảng |

---

## Hạn chế (ngắn)

> **Hạn chế.** So sánh với FGAT dùng split và tiêu chí chọn checkpoint khác nhau. Chúng tôi báo cáo validation tại epoch xếp hạng tốt nhất, trong khi baseline dùng test. So sánh công bằng tuyệt đối cần thống nhất cách xây đồ thị, negative sampling và quy tắc checkpoint.
