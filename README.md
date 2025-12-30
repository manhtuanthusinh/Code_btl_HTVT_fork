## pipeline mô phỏng định tuyến QoS-aware trong mạng IoT 25 node, gồm 4 khối chính:

### Xây dựng mô hình mạng IoT -> Cài đặt các thuật toán định tuyến -> Mô phỏng và thu thập chỉ số QoS -> Trực quan hóa kết quả

## Chức năng các file trong pipeline:
### 1. Model_graph_25node.py 
#### (Khởi tạo hạ tầng): Xây dựng mô hình mạng IoT dưới dạng đồ thị: 

    Tạo vị trí các nút mạng ngẫu nhiên và thiết lập liên kết dựa trên bán kính truyền thông.

    Tính toán các tham số QoS cho mỗi cạnh: Độ trễ (delay), Năng lượng tiêu thụ (energy), và Tỷ lệ giao gói thành công (PDR).
