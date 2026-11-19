# PkmLens

Website đồ án Học Máy: **FastAPI + React 18.3.1 + Vite**, camera live dùng YOLO26, so khớp DINOv2/FAISS và OCR. Kèm cửa sổ **PkmLens Studio** để tự chuẩn bị dataset và train.

## Mở trên máy hiện tại

- **Website toàn diện (Local + Cloudflare Tunnel)**: chạy `start-all.bat`. Script tự động khởi động Backend (`:8100`), AI Service (`:8101`), Frontend Vite (`:5173`) và tự động bật **Cloudflare Tunnel** trỏ ra ngoài Internet (`https://xxxx.trycloudflare.com`) để kết nối với Vercel hoặc truy cập từ điện thoại/máy tính khác.
- **Website local riêng**: chạy `start-pkmlens.bat`, mở **http://127.0.0.1:5173**.
- **Dừng hệ thống**: chạy `stop-pkmlens.bat` (tự động dọn dẹp các tiến trình backend và đóng cửa sổ Cloudflare Tunnel).
- **Công cụ train**: chạy `start-trainer.bat`.
- Script ưu tiên `.venv` trong PkmLens; khi chưa tạo, dùng môi trường CUDA đã có ở `D:\Project\CardLens\.venv-yolo`.
- Backend PkmLens chạy cổng **8100**, AI **8101**. Database riêng: `data/pkmlens.db`; cookie riêng `pkmlens_session`. Không dùng tài khoản/database của CardLens.
- Đăng ký tài khoản trên web để đăng bán. Mật khẩu ít nhất 12 ký tự. Tin đăng cần ảnh mặt trước và mặt sau. Giỏ hàng dùng số dư mô phỏng, không thu tiền thật.

## Triển khai Frontend lên Vercel & Kết nối Backend

Frontend hỗ trợ deploy trực tiếp lên Vercel dưới dạng Static Site (SPA):
1. **Repository**: Đẩy source code lên GitHub và Import vào Vercel (chọn Framework Preset là **Vite**).
2. **Biến môi trường trên Vercel** (`Settings` -> `Environment Variables`):
   - Thêm `VITE_API_URL`: gán bằng URL tunnel public trỏ tới Backend Python (ví dụ URL `https://xxxx.trycloudflare.com` do `start-all.bat` tạo ra, không có dấu `/` ở cuối).
   - *(Tùy chọn)* `VITE_WS_URL`: WebSocket URL (mặc định frontend tự chuyển `http(s)` của `VITE_API_URL` sang `ws(s)` và gắn path `/api/v1/ws/scan`).
3. **Redeploy**: Sau khi đổi biến môi trường, vào tab Deployments chọn **Redeploy** để Vite build lại bundle.
4. **CORS & Tunnel Bypass**: Backend FastAPI đã tích hợp sẵn CORS cho các domain `*.vercel.app`, đồng thời frontend tự động đính kèm header bypass tunnel để tương thích hoàn toàn với Cloudflare Tunnel, localtunnel, ngrok.

## Cài môi trường riêng

Python 3.12 có tkinter, Node.js 20.19+ hoặc 22.12+.

```powershell
# CPU
.\scripts\setup.ps1

# Hoặc NVIDIA CUDA 12.6
.\scripts\setup.ps1 -Cuda

.\scripts\dev.ps1
```

Các weight và dữ liệu lớn ở `data/`, ngoài Git. Thư mục này đã được chuẩn bị trên máy hiện tại; bản chỉ có source cần thêm ảnh, gallery/index DINOv2 và weight detector trước khi quét. `scripts/prepare.py` kiểm tra checksum và tạo database từ các JSON đã tải, không tạo tài khoản/tin đăng giả.

## Tự train bằng giao diện

1. Mở **PkmLens Studio** bằng `start-trainer.bat`.
2. Tab **Dữ liệu**: bộ `data/datasets/cards-synthetic-v2` đã sẵn sàng. Có thể tạo bộ mới từ thư mục ảnh full-card hoặc nhập `data.yaml`/`dataset.yaml` của bộ YOLO một lớp.
3. Tab **Huấn luyện** giữ luồng detector bbox hiện có. Với keypoint 4 góc, dùng tab Pose riêng bên dưới.
4. Xem log thật và tab **Kết quả & model**. Nút dừng giữ checkpoint sau epoch; **resume.pt** giữ optimizer để tiếp tục, kể cả khi Ultralytics rút gọn `last.pt`.
5. Chọn thư mục run rồi **Dùng model này trong website**. Model phải qua cổng validation. Dừng camera, tải lại trang quét để nạp danh sách model mới, sau đó bật lại camera.

`best.pt`: weight tốt nhất để inference. `resume.pt`: checkpoint đủ trạng thái để tiếp tục một run chưa hoàn tất. Run đã hoàn tất số epochs dự kiến cần bắt đầu một run fine-tune mới, không gọi resume như một run bị ngắt.

ONNX/TensorRT status and parity evidence: see reports/onnx-pose-verification.json. Export does not mean deployment support.


### Pose · 4 góc

- Tab Pose riêng dùng `data/datasets/cards-pose-synthetic-v1` (cảnh synthetic có corner labels) hoặc manifest góc đã được review. Chọn checkpoint local `data/models/card-detector-best.pt`; model YAML tạo head YOLO26n/s `[4,3]`, transfer các weights tương thích. Không tải weight ngầm.
- Thiết lập n/s, epochs, batch, image size và CPU/GPU; Stop giữ checkpoint, resume tiếp tục đúng dataset/config bằng `resume.pt`.
- Tab Kết quả có evaluate held-out test riêng (mặc định), val tùy chọn, lưu thư mục mới; test không dùng để chọn checkpoint.
- Checkpoint đã đăng ký `pose-yolo26n-synthetic-v1` chỉ được dùng ở chế độ experimental, opt-in; bbox là mặc định. Mọi metric hiện có đều synthetic, chưa xác nhận camera vật lý.
- Xem `reports/verification.md` và `reports/onnx-pose-verification.json` để đọc kết quả train, test, camera giả lập, ONNX và các giới hạn.
## Dữ liệu và giới hạn AI

- Catalog SQLite (`data/pkmlens.db`) hiện có **67.243 bản in** từ CardCorpus (67.085 ảnh mặt trước, 158 bản in thiếu ảnh, 104.103 tùy chọn biến thể nguồn; snapshot SHA-256 `0203e5097e020129ab76e398f5c800e2f58c1536e246b455b2a375468f43b51b`). Paired gallery bundle `data/models/cardcorpus-scanner` (`mapping.json` SHA-256 `43c31696e18dbe08f5dac35540ad3dbe51bd462478c9c7fc9eaee3a69005eed1`) nối **102.185 ảnh tham chiếu** (99.173 ảnh mapped, **65.297 bản in được đánh chỉ mục**, **101.902 / 104.519 biến thể catalog** tính cả 464 mã biến thể chỉ có ở nguồn); chế độ gallery chọn qua `PKMLENS_GALLERY_BACKEND` (`auto` mặc định, `cardcorpus` hoặc `legacy`), trong khi `ml/benchmark.py` giữ `--backend legacy` làm mặc định và hỗ trợ `--backend cardcorpus`. Bộ legacy **127 ảnh tham chiếu** (`data/gallery` + `data/metadata.csv` với các cột `image_path,card_id,name,set,number,rarity`) vẫn được giữ nguyên.
- Dataset v2 gồm 600 train, 100 validation, 100 test, nhãn bbox YOLO tự sinh. Tách theo ảnh nguồn/artwork CardCorpus để giảm rò rỉ. Đây là **cảnh tổng hợp**, không phải ảnh chụp thực tế.
- Detector ban đầu lấy từ checkpoint CardLens, ghi rõ nguồn và checksum. Không đổi nhãn COCO thành thẻ.
- YOLO tìm vị trí; OCR và gallery tìm `card_id`. Thêm ảnh thẻ mới rồi xây lại gallery không đòi hỏi thêm lớp card-ID vào YOLO.
- Thẻ hoàn toàn vắng mặt trong catalog không được tự tạo mã hay giá. OCR có thể đọc tên/số, kết quả yếu cần xác nhận thủ công.
- Màn hình camera cho chọn lại một trong ba ứng viên, tìm bản in khác trong catalog hoặc lưu văn bản OCR ở dạng chưa định danh. CSV giữ trạng thái này và để trống `card_id` của OCR chưa xác minh.
- Giá là dữ liệu tham khảo theo biến thể/tình trạng, tách giá người bán. Có nguồn/thời điểm; không có giá thì hiện thiếu. CardCorpus metadata không phải bảng giá. Hệ số tình trạng kế thừa là giả định thực nghiệm, chưa được hiệu chỉnh thị trường.
- Camera thật cần localhost hoặc HTTPS. Truy cập điện thoại qua IP HTTP sẽ không cấp camera.

- Live pose is opt-in; bbox remains the default. The current pose checkpoint is experimental because physical-camera evaluation is unavailable; see reports/verification.md.
- Pose data/train/evaluate/activation instructions and tested synthetic results are in reports/verification.md.

## Kiểm tra

```powershell
$env:PYTHONPATH="services/core-api;services/ai;."
.venv\Scripts\python.exe -m pytest -q
npm.cmd run typecheck
npm.cmd run build
```

Xem `STATUS.md` và `reports/verification.md` để biết kết quả đã đo và những phần chưa kiểm chứng. Không lấy mAP ảnh tổng hợp hoặc luồng video camera giả lập làm độ chính xác webcam thật.

Các kiểm tra trình duyệt có thể chạy khi website đang mở:

```powershell
node scripts/browser-visual.cjs
node scripts/browser-camera.cjs
node scripts/browser-smoke.cjs
```

Chúng dùng Playwright cài sẵn tại CardLens trên máy hiện tại; có thể đặt `PLAYWRIGHT_MODULE`/`CHROME_PATH` nếu dùng vị trí khác. Bài smoke tạo tài khoản kiểm thử riêng trong PkmLens. Chạy lại cùng ảnh sẽ kiểm tra nhánh kiểm duyệt ảnh trùng thay vì tự bỏ qua kiểm duyệt.

## Nguồn

- [CardCorpus dataset](https://github.com/cardcorpus/pokemon-card-dataset), CC BY 4.0 cho dữ liệu; hình ảnh/nhãn hiệu có quyền riêng của chủ sở hữu.
- [TCGdex](https://tcgdex.dev/) cho ảnh/thông tin và bản lưu giá sẵn có.
- [PokéBooth](https://pokebooth.com/#/market) là tham chiếu giao diện do chủ dự án cung cấp, không sao chép thương hiệu/người bán/đánh giá.
- Phần backend và công cụ ML dùng lại từ source local `D:\Project\CardLens`. Không sửa dự án đó.
- Live scan thresholds are adjustable through `PKMLENS_RECOGNITION_MIN_SIMILARITY`, `PKMLENS_OPEN_SET_MIN_SIMILARITY`, `PKMLENS_RECOGNITION_MIN_MARGIN`, `PKMLENS_RECOGNITION_CONSENSUS_FRAMES`, `PKMLENS_IMAGE_MIN_SHARPNESS`, `PKMLENS_IMAGE_MIN_BRIGHTNESS`, `PKMLENS_IMAGE_MAX_GLARE`, `PKMLENS_CARD_CHANGE_RATIO`, `PKMLENS_TRACK_SIGNATURE_TIMEOUT_SECONDS`, `PKMLENS_RECOGNITION_INTERVAL_SECONDS`, `PKMLENS_BEST_FRAME_IMPROVEMENT_RATIO`, and `PKMLENS_UNCONFIRMED_REFRESH_SECONDS`. The AI service validates them at startup; restart it after changes. Defaults are conservative and not calibrated on independent webcam photos.
- Track đã consensus không chạy lại OCR/embedding liên tục; thay đổi card signature sẽ reset identity và cho phép nhận diện lại.
- Pose dataset export dùng class `pokemon_card` duy nhất cho cả hai mặt; `front/back` vẫn nằm trong manifest, bốn góc giữ thứ tự TL/TR/BR/BL.
- Benchmark xuất confusion matrix theo từng phương pháp, có unknown/reject; chỉ phản ánh manifest test được cung cấp.
Video limitation: current bounded synthetic MP4 testing did not reach identity consensus within 30 seconds; do not infer video accuracy from the successful still-image or fake-camera test. See `reports/browser-ai-media-verification.json`.
