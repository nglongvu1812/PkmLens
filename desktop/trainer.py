"""Local training workspace; long-running work stays in child processes."""
import csv
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from uuid import uuid4
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'services/core-api'), str(ROOT / 'services/ai'), str(ROOT)]
os.environ.setdefault('DATA_DIR', str(ROOT / 'data'))
os.environ.setdefault('DATABASE_URL', 'sqlite:///' + (ROOT / 'data/pkmlens.db').as_posix())
os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'services/core-api'), str(ROOT / 'services/ai'), str(ROOT)])
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'data/yolo-config'))


def pose_command(dataset, checkpoint, output, scale='n', epochs=25, batch=2, imgsz=480, device='auto', resume=False):
    dataset, checkpoint, output = Path(dataset).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    if not (dataset / 'pose.yaml').is_file():
        raise ValueError('Chọn thư mục dataset có pose.yaml.')
    if checkpoint.suffix.lower() != '.pt' or not checkpoint.is_file():
        raise ValueError('Chọn checkpoint .pt cục bộ. Studio không tải model.')
    epochs, batch, imgsz = int(epochs), int(batch), int(imgsz)
    if scale not in ('n', 's') or epochs not in range(1, 301) or batch not in range(1, 17) or imgsz not in (320, 480, 640) or device not in ('auto', 'cpu', '0'):
        raise ValueError('Pose cần scale n/s, epochs từ 1 đến 300, batch từ 1 đến 16, ảnh 320/480/640 và thiết bị auto/cpu/0.')
    command = ['-m', 'ml.train_pose', 'train', '--data', str(dataset / 'pose.yaml'),
               '--checkpoint', str(checkpoint), '--output', str(output), '--scale', scale,
               '--epochs', str(epochs), '--batch', str(batch), '--imgsz', str(imgsz), '--device', device]
    if resume:
        command.append('--resume')
    return command


def pose_resume_settings(filename):
    checkpoint = Path(filename).resolve()
    if checkpoint.name == 'last.pt' and (checkpoint.parent / 'resume.pt').is_file():
        checkpoint = checkpoint.parent / 'resume.pt'
    if (checkpoint.parent.name != 'weights' or not checkpoint.is_file() or checkpoint.suffix != '.pt' or
            not (checkpoint.stem in ('resume', 'last') or checkpoint.stem.startswith('epoch') and checkpoint.stem[5:].isdigit())):
        raise ValueError('Chọn weights/resume.pt, last.pt hoặc epoch checkpoint của run Pose.')
    output = checkpoint.parents[1]
    try:
        config = json.loads((output / 'pose-config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Run cần pose-config.json để resume đúng cấu hình.') from exc
    if not isinstance(config, dict) or not config.get('dataset') or not Path(config['dataset']).is_absolute():
        raise ValueError('pose-config.json cần dataset là đường dẫn thư mục tuyệt đối.')
    if not Path(config['dataset']).is_dir() or not config.get('checkpoint'):
        raise ValueError('Thiếu thư mục dataset hoặc checkpoint khởi tạo trong pose-config.json.')
    settings = {key: config.get(key, default) for key, default in
                [('scale', 'n'), ('epochs', 25), ('batch', 2), ('imgsz', 480), ('device', 'auto')]}
    command = pose_command(config['dataset'], checkpoint, output, resume=True, **settings)
    return output, config, command


class Trainer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PkmLens Studio · Dữ liệu & huấn luyện YOLO')
        self.geometry('1120x780')
        self.minsize(940, 680)
        self.configure(bg='#fcf9f4')
        self.proc = None
        self.output = None
        self.active_run = None
        self.output_kind = 'bbox'
        self.job_id = 0
        self.pending_job = None
        self.close_when_finished = False
        self.messages = queue.Queue()
        self.dataset = tk.StringVar(value=str(ROOT / 'data/datasets/cards-synthetic-v2'))
        self.gallery = tk.StringVar(value=str(ROOT / 'data/gallery'))
        self.epochs = tk.IntVar(value=25)
        self.batch = tk.IntVar(value=4)
        self.imgsz = tk.StringVar(value='480')
        self.device = tk.StringVar(value='auto')
        self.model = tk.StringVar(value='n')
        self.scene_count = tk.IntVar(value=600)
        self.pose_dataset = tk.StringVar(value='')
        initializer = ROOT / 'data/models/card-detector-best.pt'
        self.pose_checkpoint = tk.StringVar(value=str(initializer) if initializer.is_file() else '')
        self.pose_scale = tk.StringVar(value='n')
        self.pose_epochs = tk.IntVar(value=25)
        self.pose_batch = tk.IntVar(value=2)
        self.pose_imgsz = tk.StringVar(value='480')
        self.pose_device = tk.StringVar(value='auto')
        self.pose_eval_split = tk.StringVar(value='test')
        self.progress = tk.StringVar(value='Sẵn sàng. Chọn dataset và kiểm tra trước khi train.')
        self.gpu_info = tk.StringVar(value='Đang kiểm tra thiết bị…')
        self.metric_info = tk.StringVar(value='Chưa có kết quả train trong phiên này.')
        self.history = []
        self.metric_timestamp = None
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('.', font=('Segoe UI', 10), background='#fcf9f4', foreground='#302e29')
        self.style.configure('TFrame', background='#fcf9f4')
        self.style.configure('TLabel', background='#fcf9f4')
        self.style.configure('TNotebook', background='#fcf9f4', borderwidth=0)
        self.style.configure('TNotebook.Tab', padding=(22, 12), background='#f0e8dc')
        self.style.map('TNotebook.Tab', background=[('selected', '#fff')], foreground=[('selected', '#b84c18')])
        self.style.configure('TButton', padding=(13, 9), background='#fff', bordercolor='#e3d7c8')
        self.style.configure('Primary.TButton', background='#30332e', foreground='#fff', padding=(18, 11))
        self.style.map('Primary.TButton', background=[('active', '#575245')])
        self.style.configure('TLabelframe', background='#fcf9f4', bordercolor='#e4d9c9')
        self.style.configure('TLabelframe.Label', foreground='#8b5c32', background='#fcf9f4', font=('Segoe UI', 11, 'bold'))
        self.style.configure('TEntry', fieldbackground='#fff', padding=7)
        self.build()
        self.after(150, self.drain)
        self.protocol('WM_DELETE_WINDOW', self.close)
        threading.Thread(target=self.hardware, daemon=True).start()

    def build(self):
        header = ttk.Frame(self, padding=(28, 22))
        header.pack(fill='x')
        ttk.Label(header, text='pkmlens. studio', font=('Segoe UI', 25, 'bold'), foreground='#b54c1c').pack(side='left')
        ttk.Label(header, text='Tự chuẩn bị dữ liệu. Tự train mô hình.', foreground='#85725c').pack(side='right', pady=11)
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=25)
        prepare = self.scroll_tab(notebook, 'Dữ liệu')
        train = self.scroll_tab(notebook, 'Huấn luyện')
        pose = self.scroll_tab(notebook, 'Pose · 4 góc')
        results = self.scroll_tab(notebook, 'Kết quả & model')
        self.build_pose(pose)
        self.path_row(prepare, 'Ảnh tham chiếu', self.gallery, self.choose_gallery)
        count = len(list((ROOT / 'data/gallery').glob('*.webp')))
        ttk.Label(prepare, text=f'{count} ảnh đã chuẩn bị · CardCorpus & TCGdex · metadata.csv giữ mã thẻ, set và nguồn.', foreground='#82705a').pack(anchor='w', pady=(0, 18))
        preparation = ttk.LabelFrame(prepare, text='Tạo bộ YOLO có nhãn tự động', padding=18)
        preparation.pack(fill='x', pady=8)
        ttk.Label(preparation, text='Ghép thẻ lên cảnh nền và sinh bounding box. Không cần vẽ nhãn tay.').pack(anchor='w')
        row = ttk.Frame(preparation)
        row.pack(fill='x', pady=(14, 0))
        ttk.Label(row, text='Số cảnh train').pack(side='left')
        ttk.Spinbox(row, from_=30, to=5000, textvariable=self.scene_count, width=8).pack(side='left', padx=12)
        ttk.Button(row, text='Chuẩn bị dataset', style='Primary.TButton', command=self.generate).pack(side='right')
        ttk.Label(preparation, text='Validation và test được chia theo ảnh nguồn. Số đo trên ảnh tổng hợp không phải độ chính xác camera thật.', wraplength=920, foreground='#8d724e').pack(anchor='w', pady=(13, 0))
        imports = ttk.LabelFrame(prepare, text='Hoặc dùng dataset của bạn', padding=18)
        imports.pack(fill='x', pady=13)
        ttk.Label(imports, text='Nhập data.yaml / dataset.yaml của YOLO một lớp. Dữ liệu gốc được giữ nguyên.').pack(anchor='w')
        ttk.Button(imports, text='Chọn file YAML và nhập dataset…', command=self.import_data).pack(anchor='w', pady=(12, 0))
        ttk.Button(prepare, text='Tải lại ảnh CardCorpus & xây gallery', command=self.prepare_catalog).pack(anchor='w', pady=12)
        ttk.Label(prepare, text='Dataset gán nhãn chỉ giúp phát hiện lá bài. Tên/mã thẻ được xác định qua gallery ảnh và OCR.', foreground='#82705a', wraplength=950).pack(anchor='w', pady=8)

        self.path_row(train, 'Dataset hiện tại', self.dataset, self.choose_dataset)
        ttk.Button(train, text='Kiểm tra ảnh, nhãn và các split', command=self.validate).pack(anchor='w', pady=(0, 18))
        settings = ttk.LabelFrame(train, text='Cấu hình train', padding=18)
        settings.pack(fill='x')
        for column, (label, variable, values) in enumerate([
            ('Model', self.model, ('n', 's', 'm')), ('Thiết bị', self.device, ('auto', '0', 'cpu')),
            ('Kích thước ảnh', self.imgsz, ('320', '480', '640')),
        ]):
            ttk.Label(settings, text=label).grid(row=0, column=column, sticky='w', padx=8, pady=(0, 9))
            ttk.Combobox(settings, textvariable=variable, values=values, state='readonly', width=19).grid(row=1, column=column, sticky='ew', padx=8)
            settings.columnconfigure(column, weight=1)
        for column, (label, variable, maximum) in enumerate([('Epochs', self.epochs, 300), ('Batch', self.batch, 16)]):
            ttk.Label(settings, text=label).grid(row=2, column=column, sticky='w', padx=8, pady=(18, 8))
            ttk.Spinbox(settings, from_=1, to=maximum, textvariable=variable, width=19).grid(row=3, column=column, sticky='ew', padx=8)
        ttk.Label(train, textvariable=self.gpu_info, foreground='#82705a').pack(anchor='w', pady=15)
        actions = ttk.Frame(train)
        actions.pack(fill='x', pady=7)
        ttk.Button(actions, text='Bắt đầu train', style='Primary.TButton', command=self.start_train).pack(side='left', padx=(0, 12))
        ttk.Button(actions, text='Dừng sau epoch · giữ checkpoint', command=self.stop_train).pack(side='left', padx=(0, 12))
        ttk.Button(actions, text='Resume từ last.pt…', command=self.resume).pack(side='left')
        ttk.Label(train, text='YOLO26n + batch 4 phù hợp để bắt đầu trên GTX 1660 Ti 6 GB. Giảm batch nếu thiếu VRAM.', foreground='#8d724e', wraplength=900).pack(anchor='w', pady=15)

        ttk.Label(results, textvariable=self.metric_info, foreground='#6e583f', wraplength=950).pack(anchor='w', pady=(0, 14))
        self.chart = tk.Canvas(results, height=205, bg='#fffdf8', highlightbackground='#e4d9c9', highlightthickness=1)
        self.chart.pack(fill='x', pady=(0, 15))
        self.chart.bind('<Configure>', lambda event: self.draw_chart())
        actions = ttk.Frame(results)
        actions.pack(fill='x')
        ttk.Button(actions, text='Mở kết quả một lần train…', command=self.select_run).pack(side='left', padx=(0, 12))
        ttk.Button(actions, text='Dùng model này trong website', style='Primary.TButton', command=self.activate).pack(side='left', padx=(0, 12))
        ttk.Button(actions, text='Xuất ONNX', command=self.export).pack(side='left')
        evaluation = ttk.Frame(results)
        evaluation.pack(fill='x', pady=(12, 0))
        ttk.Label(evaluation, text='Đánh giá Pose · split').pack(side='left', padx=(0, 10))
        ttk.Combobox(evaluation, textvariable=self.pose_eval_split, values=('test', 'val'), state='readonly', width=8).pack(side='left', padx=(0, 12))
        ttk.Button(evaluation, text='Đánh giá Pose đã chọn', command=self.evaluate_pose).pack(side='left')
        ttk.Label(results, text='test: đánh giá cuối trên tập giữ riêng · val: validation để chọn model. Kết quả lưu riêng, giữ nguyên gate validation.',
                  foreground='#82705a', wraplength=850).pack(anchor='w', pady=(8, 0))
        ttk.Label(results, text='Chỉ kích hoạt khi model qua validation. Model train từ cảnh tổng hợp được ghi rõ là thử nghiệm.', foreground='#8d724e', wraplength=950).pack(anchor='w', pady=16)

        bottom = ttk.Frame(self, padding=(25, 14))
        bottom.pack(side='bottom', fill='x', before=notebook)
        ttk.Label(bottom, textvariable=self.progress, foreground='#8e633b', wraplength=1020).pack(anchor='w', pady=(0, 8))
        self.log = tk.Text(bottom, height=7, background='#282e29', foreground='#e5e7da', font=('Consolas', 9), relief='flat', padx=14, pady=10, wrap='word')
        self.log.pack(fill='x')
        self.log.configure(state='disabled')
        self.append('PkmLens Studio sẵn sàng. Chuẩn bị dữ liệu hoặc mở tab Huấn luyện.\n')

    def scroll_tab(self, notebook, title):
        page = ttk.Frame(notebook)
        notebook.add(page, text=title)
        canvas = tk.Canvas(page, background='#fcf9f4', highlightthickness=0, height=380)
        scroll = ttk.Scrollbar(page, orient='vertical', command=canvas.yview)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        content = ttk.Frame(canvas, padding=24)
        window = canvas.create_window(0, 0, anchor='nw', window=content)
        content.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
        def reveal(event):
            if event.widget.winfo_toplevel() != self: return
            child = event.widget
            while child is not None and child is not content:
                child = child.master
            if child is None: return
            top = event.widget.winfo_rooty() - content.winfo_rooty()
            height = max(1, content.winfo_height())
            start = canvas.canvasy(0)
            if top < start:
                canvas.yview_moveto(top / height)
            elif top + event.widget.winfo_height() > start + canvas.winfo_height():
                canvas.yview_moveto((top + event.widget.winfo_height() - canvas.winfo_height()) / height)
        self.bind('<FocusIn>', reveal, add='+')
        return content

    def build_pose(self, parent):
        ttk.Label(parent, text='Huấn luyện bốn góc TL / TR / BR / BL. Kết quả và biểu đồ nằm ở tab Kết quả & model.',
                  wraplength=920, foreground='#82705a').pack(anchor='w', pady=(0, 14))
        columns = ttk.Frame(parent)
        columns.pack(fill='both', expand=True)
        data = ttk.LabelFrame(columns, text='Dataset & checkpoint cục bộ', padding=16)
        data.pack(side='left', fill='both', expand=True, padx=(0, 12))
        settings = ttk.LabelFrame(columns, text='Cấu hình Pose', padding=16)
        settings.pack(side='left', fill='both', expand=True)
        self.path_row(data, 'Dataset có pose.yaml', self.pose_dataset, self.choose_pose_dataset)
        self.path_row(data, 'Checkpoint khởi tạo .pt', self.pose_checkpoint, self.choose_pose_checkpoint, 'Chọn file…')
        ttk.Button(data, text='Chuyển cảnh bbox thành dataset Pose…', command=self.prepare_pose_scenes).pack(fill='x', pady=(0, 7))
        ttk.Button(data, text='Nhập manifest JSONL đã duyệt…', command=self.import_pose_manifest).pack(fill='x', pady=(0, 7))
        ttk.Button(data, text='Kiểm tra ảnh, góc và các split', command=self.validate_pose).pack(fill='x')
        ttk.Label(data, text='Tạo thư mục mới, giữ nguyên nguồn. Dữ liệu tổng hợp chưa chứng minh độ chính xác camera thật.',
                  wraplength=405, foreground='#8d724e').pack(anchor='w', pady=(12, 0))
        for row, entries in enumerate([
            [('Scale', self.pose_scale, ('n', 's')), ('Thiết bị', self.pose_device, ('auto', '0', 'cpu'))],
            [('Kích thước ảnh', self.pose_imgsz, ('320', '480', '640'))],
        ]):
            for column, (label, variable, values) in enumerate(entries):
                ttk.Label(settings, text=label).grid(row=row * 2, column=column, sticky='w', padx=5, pady=(0, 7))
                ttk.Combobox(settings, textvariable=variable, values=values, state='readonly', width=14).grid(row=row * 2 + 1, column=column, sticky='ew', padx=5, pady=(0, 12))
                settings.columnconfigure(column, weight=1)
        for column, (label, variable, maximum) in enumerate([('Epochs', self.pose_epochs, 300), ('Batch', self.pose_batch, 16)]):
            ttk.Label(settings, text=label).grid(row=4, column=column, sticky='w', padx=5, pady=(0, 7))
            ttk.Spinbox(settings, from_=1, to=maximum, textvariable=variable, width=14).grid(row=5, column=column, sticky='ew', padx=5, pady=(0, 12))
        ttk.Button(settings, text='Bắt đầu train Pose', style='Primary.TButton', command=self.start_pose_train).grid(row=6, column=0, columnspan=2, sticky='ew', pady=(0, 7))
        ttk.Button(settings, text='Dừng sau epoch · giữ checkpoint', command=self.stop_train).grid(row=7, column=0, columnspan=2, sticky='ew', pady=(0, 7))
        ttk.Button(settings, text='Resume Pose từ weights/resume.pt…', command=self.resume_pose).grid(row=8, column=0, columnspan=2, sticky='ew')
        ttk.Label(settings, textvariable=self.gpu_info, wraplength=390, foreground='#82705a').grid(row=9, column=0, columnspan=2, sticky='w', pady=(12, 0))

    def path_row(self, parent, label, variable, chooser, button_text='Chọn thư mục…'):
        ttk.Label(parent, text=label, font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 8))
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=(0, 14))
        ttk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text=button_text, command=chooser).pack(side='right', padx=(12, 0))

    def hardware(self):
        try:
            import torch
            text = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU · CUDA chưa sẵn sàng'
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                text += f' · VRAM trống {free // 1024**2:,}/{total // 1024**2:,} MB'
            self.messages.put(('gpu', text))
        except Exception as exc:
            self.messages.put(('gpu', 'Không đọc được thiết bị: ' + str(exc)))

    def choose_gallery(self):
        path = filedialog.askdirectory(title='Chọn thư mục ảnh full-card')
        if path: self.gallery.set(path)

    def choose_dataset(self):
        path = filedialog.askdirectory(title='Chọn dataset có dataset.yaml')
        if path: self.dataset.set(path)

    def busy(self):
        return self.pending_job is not None or self.proc is not None and self.proc.poll() is None

    def background(self, work, completed, current=None):
        if self.busy():
            messagebox.showinfo('Đang chạy', 'Hãy đợi hoặc dừng công việc hiện tại trước.')
            return False
        self.job_id += 1
        job = self.job_id
        self.pending_job = job
        self.progress.set('Đang kiểm tra và xử lý. Xem log bên dưới…')
        def run():
            try:
                result, error = work(), None
            except Exception as exc:
                result, error = None, str(getattr(exc, 'detail', exc))
            self.messages.put(('task', (job, result, error, completed, current)))
        threading.Thread(target=run, daemon=True).start()
        return True

    def launch(self, args, output=None, completed=None):
        if self.busy():
            messagebox.showinfo('Đang chạy', 'Hãy đợi hoặc dừng công việc hiện tại trước.')
            return False
        run = Path(output).resolve() if output else None
        training = args[:2] == ['-m', 'ml.train_detection'] or args[:3] == ['-m', 'ml.train_pose', 'train']
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}
        try:
            process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            messagebox.showerror('Không khởi chạy được', str(exc))
            return False
        self.proc = process
        self.job_id += 1
        job = self.job_id
        self.active_run = run if training else None
        def read():
            try:
                for line in process.stdout:
                    self.messages.put(('process_log', (job, line)))
            except Exception as exc:
                self.messages.put(('process_log', (job, 'Không đọc được log: ' + str(exc) + '\n')))
            finally:
                code = process.wait()
                self.messages.put(('complete', (job, process, code, completed)))
        threading.Thread(target=read, daemon=True).start()
        if run:
            self.set_output(run, 'pose' if args[:3] == ['-m', 'ml.train_pose', 'train'] else None)
        self.progress.set('Đang chạy. Xem log bên dưới…')
        return True

    def generate(self):
        try:
            count = self.scene_count.get()
            if not 30 <= count <= 5000: raise ValueError('Số cảnh train cần từ 30 đến 5000.')
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Kiểm tra số cảnh', str(exc)); return
        output = ROOT / 'data/datasets' / ('cards-' + time.strftime('%Y%m%d-%H%M%S'))
        self.launch(['-m', 'ml.detection_dataset', str(output), '--gallery', self.gallery.get(), '--count', str(count)],
                    completed=lambda: self.dataset.set(str(output)))

    def import_data(self):
        path = filedialog.askopenfilename(title='Chọn cấu hình YOLO', filetypes=[('YOLO YAML', '*.yaml *.yml')])
        if not path: return
        output = ROOT / 'data/datasets' / ('import-' + time.strftime('%Y%m%d-%H%M%S'))
        self.launch(['-m', 'ml.import_yolo', path, str(output)], completed=lambda: self.dataset.set(str(output)))

    def prepare_catalog(self):
        self.launch(['scripts/prepare.py', '--cardcorpus', '--index'])

    def validation(self, dataset):
        if not dataset:
            raise ValueError('Chọn thư mục dataset trước khi kiểm tra hoặc train.')
        from cardlens_ai.training import validate_dataset
        result = validate_dataset(Path(dataset).resolve())
        if not result['valid']:
            raise ValueError('\n'.join(result['errors']))
        return result

    def show_validation(self, result):
        self.progress.set('Dataset hợp lệ: ' + json.dumps(result['counts']))
        self.append(json.dumps(result, indent=2, ensure_ascii=False) + '\n')

    def validate(self):
        dataset = self.dataset.get()
        self.background(lambda: self.validation(dataset), self.show_validation, lambda: self.dataset.get() == dataset)

    def start_train(self):
        try:
            epochs, batch = self.epochs.get(), self.batch.get()
            if epochs not in range(1, 301) or batch not in range(1, 17):
                raise ValueError('Epochs từ 1 đến 300, batch từ 1 đến 16.')
            dataset, model, imgsz, device = self.dataset.get(), self.model.get(), self.imgsz.get(), self.device.get()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Dataset hoặc cấu hình chưa hợp lệ', str(exc)); return
        output = ROOT / 'data/training' / ('yolo26' + model + '-' + time.strftime('%Y%m%d-%H%M%S'))
        def ready(result):
            self.show_validation(result)
            self.launch(['-m', 'ml.train_detection', dataset, str(output), '--model', model,
                         '--epochs', str(epochs), '--batch', str(batch), '--imgsz', imgsz, '--device', device], output=output)
        self.background(lambda: self.validation(dataset), ready,
                        lambda: (self.dataset.get(), self.model.get(), self.epochs.get(), self.batch.get(), self.imgsz.get(), self.device.get()) ==
                        (dataset, model, epochs, batch, imgsz, device))

    def choose_pose_dataset(self):
        path = filedialog.askdirectory(title='Chọn thư mục dataset có pose.yaml', initialdir=ROOT / 'data/datasets')
        if path: self.pose_dataset.set(path)

    def choose_pose_checkpoint(self):
        path = filedialog.askopenfilename(title='Chọn checkpoint khởi tạo cục bộ · không tải model',
                                          initialdir=ROOT / 'data/models', filetypes=[('PyTorch checkpoint', '*.pt')])
        if path: self.pose_checkpoint.set(path)

    def prepare_pose_scenes(self):
        source = filedialog.askdirectory(title='Chọn cảnh bbox có manifest.jsonl', initialdir=self.dataset.get())
        if not source: return
        output = ROOT / 'data/datasets' / ('pose-scenes-' + str(time.time_ns()))
        selected = self.pose_dataset.get()
        self.launch(['-m', 'ml.pose_dataset', 'prepare', source, str(output)],
                    completed=lambda: self.pose_dataset.set(str(output)) if self.pose_dataset.get() == selected else None)

    def import_pose_manifest(self):
        manifest = filedialog.askopenfilename(title='Chọn manifest có góc TL/TR/BR/BL đã duyệt', filetypes=[('Reviewed manifest', '*.jsonl')])
        if not manifest: return
        output = ROOT / 'data/datasets' / ('pose-reviewed-' + str(time.time_ns()))
        selected = self.pose_dataset.get()
        self.launch(['-m', 'ml.prepare_pose', manifest, str(output)],
                    completed=lambda: self.pose_dataset.set(str(output)) if self.pose_dataset.get() == selected else None)

    @staticmethod
    def pose_validation(dataset):
        from ml.pose_dataset import validate_pose_dataset
        result = validate_pose_dataset(Path(dataset).resolve())
        if not result['valid']:
            raise ValueError('\n'.join(result['errors']))
        return result

    def validate_pose(self):
        dataset = self.pose_dataset.get()
        if not dataset:
            messagebox.showinfo('Chọn dataset Pose', 'Chọn thư mục có pose.yaml hoặc chuẩn bị dataset Pose trước.'); return
        self.background(lambda: self.pose_validation(dataset), self.show_validation, lambda: self.pose_dataset.get() == dataset)

    def pose_values(self):
        return (self.pose_dataset.get(), self.pose_checkpoint.get(), self.pose_scale.get(), self.pose_epochs.get(),
                self.pose_batch.get(), self.pose_imgsz.get(), self.pose_device.get())

    def start_pose_train(self):
        try:
            values = self.pose_values()
            dataset, checkpoint, scale, epochs, batch, imgsz, device = values
            if not dataset or not checkpoint:
                raise ValueError('Chọn dataset Pose và checkpoint khởi tạo .pt cục bộ trước.')
            output = ROOT / 'data/training' / ('pose-' + str(time.time_ns()))
            command = pose_command(dataset, checkpoint, output, scale, epochs, batch, imgsz, device)
        except (ValueError, TypeError, tk.TclError) as exc:
            messagebox.showerror('Kiểm tra cấu hình Pose', str(exc)); return
        def ready(result):
            self.show_validation(result)
            self.launch(command, output=output)
        self.background(lambda: self.pose_validation(dataset), ready, lambda: self.pose_values() == values)

    def resume_pose(self):
        if self.busy():
            messagebox.showinfo('Đang chạy', 'Đợi công việc hiện tại kết thúc trước khi resume.'); return
        filename = filedialog.askopenfilename(title='Resume Pose · chọn weights/resume.pt (ưu tiên) hoặc last.pt',
                                             initialdir=ROOT / 'data/training', filetypes=[('PyTorch checkpoint', '*.pt')])
        if not filename: return
        try:
            output, config, command = pose_resume_settings(filename)
        except (ValueError, TypeError, tk.TclError) as exc:
            messagebox.showerror('Không resume được Pose', str(exc)); return
        self.pose_dataset.set(config['dataset'])
        self.pose_checkpoint.set(config['checkpoint'])
        for name in ('scale', 'epochs', 'batch', 'imgsz', 'device'):
            getattr(self, 'pose_' + name).set(command[command.index('--' + name) + 1])
        values = self.pose_values()
        self.background(lambda: self.pose_validation(config['dataset']), lambda result: self.launch(command, output=output),
                        lambda: self.pose_values() == values)

    def stop_train(self):
        if self.active_run and self.proc and self.proc.poll() is None:
            try:
                self.active_run.mkdir(parents=True, exist_ok=True)
                (self.active_run / 'STOP').touch()
            except OSError as exc:
                messagebox.showerror('Không gửi được yêu cầu dừng', str(exc)); return False
            self.progress.set('Đang chờ epoch hiện tại kết thúc. Checkpoint sẽ được giữ lại.')
            return True

    def resume(self):
        if self.busy():
            messagebox.showinfo('Đang chạy', 'Đợi công việc hiện tại kết thúc trước khi resume.'); return
        file = filedialog.askopenfilename(title='Chọn last.pt hoặc epoch checkpoint', filetypes=[('PyTorch checkpoint', '*.pt')], initialdir=ROOT / 'data/training')
        if not file: return
        path = Path(file).resolve()
        if path.name == 'last.pt' and (path.parent / 'resume.pt').is_file():
            path = path.parent / 'resume.pt'
        if path.parent.name != 'weights':
            messagebox.showerror('Không phải checkpoint train', 'Chọn checkpoint bên trong thư mục weights của một run.'); return
        output = path.parents[1]
        config = self.read_json(output / 'config.json')
        if not config.get('dataset'):
            messagebox.showerror('Thiếu cấu hình', 'Run chưa có config.json với đường dẫn dataset.'); return
        (output / 'STOP').unlink(missing_ok=True)
        self.dataset.set(config['dataset'])
        self.launch(['-m', 'ml.train_detection', config['dataset'], str(output), '--resume', str(path),
                     '--epochs', str(config.get('epochs', 25)), '--device', self.device.get()], output=output)

    @staticmethod
    def run_kind(output):
        return 'pose' if any((output / name).is_file() for name in ('pose-config.json', 'pose-status.json', 'pose-result.json', 'pose-evaluation.json')) else 'bbox'

    def set_output(self, output, kind=None):
        self.output = Path(output).resolve()
        self.output_kind = kind or self.run_kind(self.output)
        self.metric_timestamp = None
        self.history = []
        self.metric_info.set('Đang đọc kết quả ' + self.output.name + '…')
        self.draw_chart()

    def select_run(self):
        path = filedialog.askdirectory(title='Chọn thư mục kết quả train bbox hoặc Pose', initialdir=ROOT / 'data/training')
        if path:
            self.set_output(path)
            self.update_metrics()

    def evaluate_pose(self):
        if self.busy():
            messagebox.showinfo('Đang chạy', 'Đợi công việc hiện tại kết thúc trước khi đánh giá.'); return
        if not self.output or self.output_kind != 'pose':
            messagebox.showinfo('Chọn kết quả Pose', 'Mở kết quả train Pose ở tab Kết quả & model trước.'); return
        run = self.output.resolve()
        try:
            config = self.read_json(run / 'pose-config.json')
            result = self.read_json(run / 'pose-result.json')
            dataset = Path(config.get('dataset') or '')
            if not dataset.is_absolute() or not (dataset / 'pose.yaml').is_file():
                raise ValueError('Run cần pose-config.json với thư mục dataset tuyệt đối có pose.yaml.')
            checkpoint = Path(result.get('checkpoint') or run / 'weights/best.pt').resolve()
            if (result.get('status') not in ('completed', 'stopped') or not checkpoint.is_relative_to(run) or
                    not checkpoint.is_file() or checkpoint.suffix.lower() != '.pt'):
                raise ValueError('Run Pose cần kết quả train hoàn tất và checkpoint .pt trong thư mục kết quả.')
            batch, imgsz, device = config.get('batch', 2), config.get('imgsz', 480), config.get('device', 'auto')
            split = self.pose_eval_split.get()
            if (type(batch) is not int or batch not in range(1, 17) or type(imgsz) is not int or
                    imgsz not in (320, 480, 640) or device not in ('auto', 'cpu', '0') or split not in ('val', 'test')):
                raise ValueError('Kiểm tra batch, imgsz, device của run và chọn split val hoặc test.')
            output = run / 'evaluations' / (split + '-' + uuid4().hex)
            command = ['-m', 'ml.train_pose', 'evaluate', '--data', str(dataset / 'pose.yaml'),
                       '--checkpoint', str(checkpoint), '--output', str(output), '--split', split,
                       '--batch', str(batch), '--imgsz', str(imgsz), '--device', device]
        except (ValueError, TypeError, OSError, tk.TclError) as exc:
            messagebox.showerror('Không đánh giá được Pose', str(exc)); return
        def done():
            report = self.read_json(output / 'pose-evaluation.json')
            if (report.get('status') != 'completed' or report.get('split') != split or
                    not isinstance(report.get('metrics'), dict) or not report['metrics']):
                raise ValueError('Chưa có pose-evaluation.json hoàn tất. Xem log và chạy lại đánh giá.')
            self.append('Đã lưu đánh giá Pose: ' + str(output / 'pose-evaluation.json') + '\n')
            if self.output == run:
                self.update_metrics()
        if self.launch(command, completed=done):
            self.progress.set('Đang đánh giá Pose · ' + ('held-out test' if split == 'test' else 'validation') + ' · ' + run.name)

    def activate(self):
        if not self.output:
            messagebox.showinfo('Chọn kết quả', 'Mở một kết quả train trước.'); return
        output, kind = self.output.resolve(), self.output_kind
        def work():
            if output.parent != (ROOT / 'data/training').resolve():
                raise ValueError('Chọn run bên trong data/training của PkmLens.')
            from cardlens_ai.training import Activate, activate, activate_pose
            return activate_pose(output.name, experimental=True) if kind == 'pose' else activate(output.name, Activate(experimental=True))
        def done(result):
            self.progress.set('Đã chọn model ' + result['active'] + '. Dừng/bật lại camera website để dùng.')
        self.background(work, done, lambda: self.output == output)

    def export(self):
        if not self.output: return
        if self.output_kind != 'pose':
            self.launch(['-m', 'ml.export_detector', str(self.output)])
            return
        result = self.read_json(self.output / 'pose-result.json')
        checkpoint = Path(result.get('checkpoint') or self.output / 'weights/best.pt').resolve()
        if not checkpoint.is_relative_to(self.output.resolve()) or not checkpoint.is_file() or checkpoint.suffix.lower() != '.pt':
            messagebox.showerror('Chưa xuất được Pose', 'Run chưa có checkpoint .pt hợp lệ trong thư mục kết quả.'); return
        self.launch(['-m', 'ml.train_pose', 'export', '--checkpoint', str(checkpoint), '--output', str(self.output)])

    @staticmethod
    def read_json(path):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError): return {}

    def update_metrics(self):
        if not self.output: return
        pose = self.output_kind == 'pose'
        prefix = 'pose-' if pose else ''
        state = self.read_json(self.output / (prefix + 'status.json'))
        result = self.read_json(self.output / (prefix + 'result.json'))
        config = self.read_json(self.output / (prefix + 'config.json'))
        if state.get('epoch') and self.output == self.active_run:
            self.progress.set('Epoch {}/{} · {}'.format(state['epoch'], state.get('epochs', '?'), self.output.name))
        try:
            path = self.output / 'results.csv'
            stat = path.stat()
            timestamp = (str(self.output), stat.st_mtime_ns, stat.st_size)
            if self.metric_timestamp != timestamp:
                history = []
                with path.open(encoding='utf-8-sig') as handle:
                    for row in csv.DictReader(handle):
                        try:
                            measured = {key.strip(): float(value) for key, value in row.items() if key and value is not None and value.strip()}
                            history.append({key: value for key, value in measured.items() if math.isfinite(value)})
                        except (ValueError, TypeError): continue
                self.history = history
                self.metric_timestamp = timestamp
        except OSError: pass
        metrics = result.get('metrics') or state.get('metrics') or (self.history[-1] if self.history else {})
        suffix = '(P)' if pose else '(B)'
        def metric_labels(measured_metrics):
            labels = []
            for label, key in [('Precision', 'precision'), ('Recall', 'recall'), ('mAP50', 'mAP50'), ('mAP50-95', 'mAP50-95')]:
                value = measured_metrics.get('metrics/' + key + suffix)
                measured = f'{value:.3f}' if isinstance(value, (int, float)) and math.isfinite(value) else 'chưa có'
                labels.append(label + suffix + ': ' + measured)
            return '   ·   '.join(labels)
        metadata = self.read_json(Path(config['dataset']) / 'dataset.json') if config.get('dataset') else {}
        kinds = result.get('validation_source_kinds') or config.get('validation_source_kinds') or []
        source = ', '.join(kinds) or result.get('dataset_kind') or config.get('dataset_kind') or metadata.get('kind') or ', '.join(metadata.get('source_kinds', [])) or 'chưa ghi nguồn'
        if source == 'synthetic_only' or kinds and all(kind.startswith('synthetic') for kind in kinds): source = 'tổng hợp'
        evaluated = result.get('status') in ('completed', 'stopped') and bool(result.get('metrics'))
        gate = ('Đạt' if result.get('validation_passed') else 'Không đạt') if evaluated else 'Chưa đánh giá'
        if result.get('status') and result['status'] != 'completed': gate += ' · ' + result['status']
        evidence = 'đã đánh giá ảnh thật' if result.get('real_photo_validated') else 'chưa xác nhận camera thật'
        split = result.get('validation_split') or result.get('split') or 'val'
        role = ('Held-out test (test)' if split == 'test' else 'Validation (val)') + ' · ' if pose else ''
        info = (('Pose · 4 góc' if pose else 'BBox') + ' · ' + self.output.name + '\n' + role + metric_labels(metrics) +
                f'\nNguồn: {source} · Split: {split} · Gate validation: {gate} · {evidence}')
        if pose:
            evaluation, latest = {}, -1
            for path in [self.output / 'pose-evaluation.json', *self.output.glob('evaluations/*/pose-evaluation.json')]:
                report = self.read_json(path)
                if (report.get('status') != 'completed' or report.get('split') not in ('val', 'test') or
                        not isinstance(report.get('metrics'), dict) or not report['metrics']):
                    continue
                try:
                    modified = path.stat().st_mtime_ns
                except OSError:
                    continue
                if modified > latest:
                    evaluation, latest = report, modified
            if evaluation:
                if not result and not state and not self.history:
                    info = 'Pose · 4 góc · ' + self.output.name
                split = evaluation['split']
                role = 'Held-out test (test)' if split == 'test' else 'Validation (val)'
                kinds = evaluation.get('validation_source_kinds') or []
                source = ', '.join(kinds) or evaluation.get('dataset_kind') or 'chưa ghi nguồn'
                if source == 'synthetic_only' or kinds and all(kind.startswith('synthetic') for kind in kinds): source = 'tổng hợp'
                info += '\nĐánh giá riêng · ' + role + ' · ' + metric_labels(evaluation['metrics'])
                info += f'\nNguồn: {source} · Split: {split} · chưa xác nhận camera thật'
        self.metric_info.set(info)
        self.draw_chart()

    def draw_chart(self):
        chart = self.chart
        chart.delete('all')
        width, height = max(500, chart.winfo_width()), max(150, chart.winfo_height())
        if not self.history:
            chart.create_text(width / 2, height / 2, text='Biểu đồ mAP, precision và recall xuất hiện khi train bắt đầu.', fill='#a58a67', font=('Segoe UI', 11)); return
        margin = 35
        for value in (0, .25, .5, .75, 1):
            y = height - margin - value * (height - margin * 2)
            chart.create_line(margin, y, width - margin, y, fill='#eee5d7')
            chart.create_text(17, y, text=str(value), fill='#9a866a', font=('Segoe UI', 8))
        suffix = '(P)' if self.output_kind == 'pose' else '(B)'
        keys = [('metrics/mAP50' + suffix, '#c36925', 'mAP50' + suffix), ('metrics/precision' + suffix, '#3f7b59', 'Precision' + suffix), ('metrics/recall' + suffix, '#5578a2', 'Recall' + suffix)]
        for index, (key, color, name) in enumerate(keys):
            points = []
            for i, row in enumerate(self.history):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value): continue
                x = margin + i * (width - 2 * margin) / max(1, len(self.history) - 1)
                y = height - margin - max(0, min(1, value)) * (height - 2 * margin)
                points.extend([x, y])
            if len(points) >= 4: chart.create_line(*points, fill=color, width=2)
            elif points: chart.create_oval(points[0] - 3, points[1] - 3, points[0] + 3, points[1] + 3, fill=color, outline=color)
            chart.create_text(margin + index * 130, 15, text=name, fill=color, anchor='w', font=('Segoe UI', 9, 'bold'))
        chart.create_text(width - margin, height - 10, text='Epoch ' + str(len(self.history)), fill='#9a866a', anchor='e', font=('Segoe UI', 8))

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        if int(self.log.index('end-1c').split('.')[0]) > 1500: self.log.delete('1.0', '500.0')
        self.log.see('end'); self.log.configure(state='disabled')

    def drain(self):
        while not self.messages.empty():
            kind, value = self.messages.get_nowait()
            if kind == 'gpu': self.gpu_info.set(value)
            elif kind == 'process_log':
                job, line = value
                if job == self.job_id: self.append(line)
            elif kind == 'task':
                job, result, error, completed, current = value
                if job != self.pending_job: continue
                self.pending_job = None
                try:
                    if current and not current():
                        self.progress.set('Lựa chọn đã thay đổi. Chạy lại thao tác với lựa chọn hiện tại.')
                        continue
                    if error:
                        self.progress.set('Công việc gặp lỗi. Kiểm tra dữ liệu rồi thử lại.')
                        messagebox.showerror('Kiểm tra dữ liệu hoặc model', error)
                    else:
                        self.progress.set('Hoàn tất.')
                        completed(result)
                except (ValueError, TypeError, tk.TclError) as exc:
                    messagebox.showerror('Lựa chọn đã thay đổi', str(exc))
            elif kind == 'complete':
                job, process, code, callback = value
                if job != self.job_id or process is not self.proc: continue
                self.proc = None
                self.active_run = None
                self.progress.set('Hoàn tất.' if code == 0 else 'Công việc gặp lỗi. Xem log để xử lý rồi thử lại.')
                if code == 0 and callback:
                    try:
                        callback()
                    except (OSError, ValueError, TypeError, tk.TclError) as exc:
                        self.progress.set('Không đọc được kết quả. Xem log và chạy lại thao tác.')
                        messagebox.showerror('Không đọc được kết quả', str(exc))
                if self.close_when_finished:
                    self.destroy()
                    return
        if self.output: self.update_metrics()
        self.after(700, self.drain)

    def close(self):
        if self.pending_job is not None:
            messagebox.showinfo('Đang xử lý', 'Đợi công việc kiểm tra hoặc chọn model hoàn tất trước khi đóng cửa sổ.'); return
        if self.proc and self.proc.poll() is None:
            if not self.active_run:
                messagebox.showinfo('Công việc đang chạy', 'Đợi công việc dữ liệu hoặc xuất model hoàn tất trước khi đóng cửa sổ.'); return
            if not messagebox.askyesno('Công việc đang chạy', 'Dừng sau epoch và đóng cửa sổ? Tiến trình train sẽ giữ checkpoint trước khi kết thúc.'): return
            if not self.stop_train(): return
            self.close_when_finished = True
            self.withdraw()
            return
        self.destroy()


if __name__ == '__main__':
    app = Trainer()
    if '--smoke' in sys.argv:
        app.update()
        assert app.winfo_width() >= 940
        app.after(700, app.destroy)
    app.mainloop()
