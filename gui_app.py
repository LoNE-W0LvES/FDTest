"""
Forgery Detection GUI Application - Enhanced with Image Viewer
Complete UI for training and inference with real-time console output
With separate progress bars for Epoch, Training, and Validation
With Image Viewer showing predictions and metrics
For PyCharm on Windows
"""
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar,
    QCheckBox, QSpinBox, QComboBox, QFileDialog, QTabWidget,
    QGroupBox, QMessageBox, QDoubleSpinBox, QSlider, QRadioButton, 
    QButtonGroup, QScrollArea, QListWidget, QSplitter
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtGui import QFont, QTextCursor, QPixmap, QImage
import torch
import numpy as np
import cv2


class GPUCheckThread(QThread):
    """Thread for checking GPU availability"""
    finished = Signal(dict)
    
    def run(self):
        result = {
            'available': False,
            'device_count': 0,
            'devices': [],
            'cuda_version': 'N/A',
            'error': None
        }
        
        try:
            result['available'] = torch.cuda.is_available()
            
            if result['available']:
                result['device_count'] = torch.cuda.device_count()
                result['cuda_version'] = torch.version.cuda
                
                for i in range(result['device_count']):
                    device_info = {
                        'name': torch.cuda.get_device_name(i),
                        'memory': torch.cuda.get_device_properties(i).total_memory / 1024**3
                    }
                    result['devices'].append(device_info)
        except Exception as e:
            result['error'] = str(e)
        
        self.finished.emit(result)


class ImagePredictionThread(QThread):
    """Thread for predicting single image"""
    finished_signal = Signal(dict)
    error_signal = Signal(str)
    
    def __init__(self, model_path, image_path, model_type, device):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.model_type = model_type
        self.device = device
    
    def run(self):
        try:
            from inference import ForgeryPredictor
            from model import create_model
            from dataset import get_valid_transforms
            import torch
            
            # Set device
            device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            model = create_model(self.model_type, pretrained=False)
            
            # Handle DataParallel
            state_dict = checkpoint['model_state_dict']
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            
            model = model.to(device)
            model.eval()
            
            # Create predictor
            predictor = ForgeryPredictor(model, device, 0.5, 0.5)
            
            # Get transform
            transform = get_valid_transforms((512, 512))
            
            # Predict
            result = predictor.predict_image(self.image_path, transform)
            
            # Load original image for display
            image = cv2.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = image.copy()
            if result['is_forged']:
                mask_colored = np.zeros_like(image)
                mask_colored[:, :, 0] = result['seg_mask'] * 255  # Red channel
                overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            result_data = {
                'is_forged': bool(result['is_forged']),
                'confidence': float(result['class_prob'][1]),
                'authentic_prob': float(result['class_prob'][0]),
                'forged_prob': float(result['class_prob'][1]),
                'original_image': image,
                'mask': result['seg_mask'],
                'overlay': overlay,
                'image_shape': image.shape,
                'mask_pixels': int(result['seg_mask'].sum()) if result['is_forged'] else 0,
                'total_pixels': int(result['seg_mask'].size),
                'val_f1': checkpoint.get('val_f1', 'N/A'),
                'val_dice': checkpoint.get('val_dice', 'N/A'),
            }
            
            self.finished_signal.emit(result_data)
            
        except Exception as e:
            import traceback
            error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)


class TrainingThread(QThread):
    """Thread for training the model"""
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    epoch_signal = Signal(int, int, dict)
    training_progress_signal = Signal(int, int)
    validation_progress_signal = Signal(int, int)
    finished_signal = Signal(bool, str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.should_stop = False
        self.current_epoch = 0

    def run(self):
        try:
            from train import validate, create_data_loaders, MetricTracker
            from model import create_model, CombinedLoss
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            import torch
            import os

            device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
            os.makedirs(self.config['output_dir'], exist_ok=True)

            train_loader, val_loader = create_data_loaders(
                data_dir=self.config['data_dir'],
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                val_split=self.config['val_split'],
                img_size=self.config['img_size'],
                device=device.type
            )

            model = create_model(self.config['model_type'], pretrained=True)
            if torch.cuda.device_count() > 1:
                import torch.nn as nn
                model = nn.DataParallel(model)
            model = model.to(device)

            criterion = CombinedLoss(class_weight=1.0, seg_weight=2.0, dice_weight=0.5, bce_weight=0.5)
            optimizer = AdamW(model.parameters(), lr=self.config['learning_rate'], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=1e-6)

            best_val_f1 = 0.0
            total_train_batches = len(train_loader)
            total_val_batches = len(val_loader)

            for epoch in range(self.config['epochs']):
                if self.should_stop:
                    break

                self.current_epoch = epoch + 1
                self.log_signal.emit(f"\n{'='*70}")
                self.log_signal.emit(f"Epoch {self.current_epoch}/{self.config['epochs']}")
                self.log_signal.emit(f"{'='*70}")

                model.train()
                metrics = MetricTracker()
                
                for batch_idx, batch in enumerate(train_loader):
                    if self.should_stop:
                        break
                        
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    masks = batch['mask'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss_dict = criterion(outputs, labels, masks)
                    loss = loss_dict['total_loss']
                    loss.backward()
                    optimizer.step()
                    
                    metrics.update(loss_dict, outputs['class_logits'], labels, outputs['seg_logits'], masks)
                    self.training_progress_signal.emit(batch_idx + 1, total_train_batches)

                train_metrics = metrics.get_metrics()

                model.eval()
                metrics = MetricTracker()
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        if self.should_stop:
                            break
                            
                        images = batch['image'].to(device)
                        labels = batch['label'].to(device)
                        masks = batch['mask'].to(device)
                        
                        outputs = model(images)
                        loss_dict = criterion(outputs, labels, masks)
                        metrics.update(loss_dict, outputs['class_logits'], labels, outputs['seg_logits'], masks)
                        self.validation_progress_signal.emit(batch_idx + 1, total_val_batches)

                val_metrics = metrics.get_metrics()
                scheduler.step()

                self.log_signal.emit(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, Dice: {train_metrics['dice']:.4f}")
                self.log_signal.emit(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, Dice: {val_metrics['dice']:.4f}")

                self.epoch_signal.emit(self.current_epoch, self.config['epochs'], val_metrics)
                self.progress_signal.emit(self.current_epoch, self.config['epochs'])

                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_f1': val_metrics['f1'],
                        'val_dice': val_metrics['dice'],
                        'model_type': self.config['model_type'],
                    }
                    torch.save(checkpoint, os.path.join(self.config['output_dir'], 'best_model.pth'))
                    self.log_signal.emit(f"✓ Saved best model (F1: {best_val_f1:.4f})")

            self.finished_signal.emit(True, "Training completed successfully!")

        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.log_signal.emit(error_msg)
            self.finished_signal.emit(False, error_msg)

    def stop(self):
        self.should_stop = True


class InferenceThread(QThread):
    """Thread for running inference"""
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finished_signal = Signal(bool, str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            from inference import generate_submission
            import io
            import contextlib

            class LogCapture:
                def __init__(self, callback):
                    self.callback = callback

                def write(self, text):
                    if text.strip():
                        self.callback(text)

                def flush(self):
                    pass

            log_capture = LogCapture(self.log_signal.emit)

            with contextlib.redirect_stdout(log_capture):
                submission_df = generate_submission(
                    model_path=self.config['model_path'],
                    test_dir=self.config['test_dir'],
                    output_path=self.config['output_path'],
                    model_type=self.config['model_type'],
                    batch_size=self.config['batch_size'],
                    img_size=self.config['img_size'],
                    device='cuda' if self.config['use_gpu'] else 'cpu',
                    classification_threshold=self.config['class_threshold'],
                    segmentation_threshold=self.config['seg_threshold'],
                    num_workers=self.config['num_workers']
                )

            self.finished_signal.emit(True, f"Inference completed! Saved to {self.config['output_path']}")

        except Exception as e:
            import traceback
            error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
            self.finished_signal.emit(False, error_msg)


class TrainingTab(QWidget):
    """Training tab UI with separate progress bars"""

    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Configuration section
        config_group = QGroupBox("Training Configuration")
        config_layout = QVBoxLayout()

        # Data directory
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Data Directory:"))
        self.data_dir_input = QLineEdit()
        self.data_dir_input.setPlaceholderText("Select folder containing train_images and train_masks")
        data_layout.addWidget(self.data_dir_input)
        self.browse_data_btn = QPushButton("Browse")
        self.browse_data_btn.clicked.connect(self.browse_data_dir)
        data_layout.addWidget(self.browse_data_btn)
        config_layout.addLayout(data_layout)

        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit("checkpoints")
        output_layout.addWidget(self.output_dir_input)
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.browse_output_btn)
        config_layout.addLayout(output_layout)

        # Model type
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Type:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "unet_resnet50 (Recommended)",
            "unet_resnet34 (Faster)",
            "unet_efficientnet_b3 (More Accurate)",
            "unet_efficientnet_b4 (Best Accuracy)"
        ])
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)

        # Parameters row 1
        params1_layout = QHBoxLayout()
        params1_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(30)
        params1_layout.addWidget(self.epochs_spin)

        params1_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        params1_layout.addWidget(self.batch_spin)

        params1_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.00001)
        params1_layout.addWidget(self.lr_spin)
        config_layout.addLayout(params1_layout)

        # Parameters row 2
        params2_layout = QHBoxLayout()
        params2_layout.addWidget(QLabel("Image Size:"))
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(128, 1024)
        self.img_size_spin.setValue(512)
        self.img_size_spin.setSingleStep(64)
        params2_layout.addWidget(self.img_size_spin)

        params2_layout.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(4)
        params2_layout.addWidget(self.workers_spin)

        params2_layout.addWidget(QLabel("Val Split:"))
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.5)
        self.val_split_spin.setValue(0.15)
        self.val_split_spin.setSingleStep(0.05)
        params2_layout.addWidget(self.val_split_spin)
        config_layout.addLayout(params2_layout)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Device selection
        device_group = QGroupBox("Device Selection")
        device_layout = QVBoxLayout()

        device_choice_layout = QHBoxLayout()
        device_choice_layout.addWidget(QLabel("Training Device:"))

        self.device_button_group = QButtonGroup()

        self.gpu_radio = QRadioButton("GPU (CUDA) - Fast")
        self.gpu_radio.setChecked(True)
        self.device_button_group.addButton(self.gpu_radio, 0)
        device_choice_layout.addWidget(self.gpu_radio)

        self.cpu_radio = QRadioButton("CPU Only - Slow")
        self.device_button_group.addButton(self.cpu_radio, 1)
        device_choice_layout.addWidget(self.cpu_radio)

        self.check_gpu_btn = QPushButton("Check GPU")
        self.check_gpu_btn.clicked.connect(self.check_gpu)
        device_choice_layout.addWidget(self.check_gpu_btn)
        device_choice_layout.addStretch()

        device_layout.addLayout(device_choice_layout)

        status_layout = QHBoxLayout()
        self.gpu_status_label = QLabel("GPU Status: Unknown")
        status_layout.addWidget(self.gpu_status_label)
        status_layout.addStretch()
        device_layout.addLayout(status_layout)

        info_label = QLabel("ℹ️ GPU is 10-50x faster than CPU")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        device_layout.addWidget(info_label)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        control_layout.addWidget(self.stop_btn)

        layout.addLayout(control_layout)

        # Progress bars group
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()

        epoch_layout = QVBoxLayout()
        self.epoch_label = QLabel("Epoch: 0/0")
        self.epoch_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        epoch_layout.addWidget(self.epoch_label)
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setRange(0, 100)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        epoch_layout.addWidget(self.epoch_progress)
        progress_layout.addLayout(epoch_layout)

        train_layout = QVBoxLayout()
        self.train_label = QLabel("Training: 0/0 batches")
        train_layout.addWidget(self.train_label)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")
        train_layout.addWidget(self.train_progress)
        progress_layout.addLayout(train_layout)

        val_layout = QVBoxLayout()
        self.val_label = QLabel("Validation: 0/0 batches")
        val_layout.addWidget(self.val_label)
        self.val_progress = QProgressBar()
        self.val_progress.setRange(0, 100)
        self.val_progress.setValue(0)
        self.val_progress.setStyleSheet("QProgressBar::chunk { background-color: #FF9800; }")
        val_layout.addWidget(self.val_progress)
        progress_layout.addLayout(val_layout)

        metrics_layout = QHBoxLayout()
        self.total_metrics_label = QLabel("Total - Epochs: 0 | Training: 0 batches | Validation: 0 batches")
        self.total_metrics_label.setFont(QFont("Arial", 9))
        self.total_metrics_label.setStyleSheet("color: #666; padding: 5px;")
        metrics_layout.addWidget(self.total_metrics_label)
        progress_layout.addLayout(metrics_layout)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Console output
        console_group = QGroupBox("Training Log")
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Consolas", 9))
        console_layout.addWidget(self.console_output)
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)

        self.setLayout(layout)

        QTimer.singleShot(500, self.check_gpu)

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_dir_input.setText(dir_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def check_gpu(self):
        self.log("Checking GPU availability...")
        self.check_gpu_btn.setEnabled(False)

        self.gpu_thread = GPUCheckThread()
        self.gpu_thread.finished.connect(self.on_gpu_check_finished)
        self.gpu_thread.start()

    def on_gpu_check_finished(self, result):
        self.check_gpu_btn.setEnabled(True)

        if result['error']:
            self.gpu_status_label.setText(f"GPU Status: Error - {result['error']}")
            self.gpu_status_label.setStyleSheet("color: red;")
            self.log(f"GPU check error: {result['error']}")
            self.gpu_radio.setEnabled(False)
            self.cpu_radio.setChecked(True)
            return

        if result['available']:
            device_info = f"{result['device_count']} GPU(s) - {result['devices'][0]['name']}"
            self.gpu_status_label.setText(f"GPU Status: ✓ {device_info}")
            self.gpu_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log(f"✓ GPU Available: {result['device_count']} device(s)")
            for i, device in enumerate(result['devices']):
                self.log(f"  GPU {i}: {device['name']} ({device['memory']:.1f} GB)")
            self.gpu_radio.setEnabled(True)
            self.gpu_radio.setChecked(True)

            if result['device_count'] > 1:
                self.log(f"✓ Multiple GPUs detected! Training will use all {result['device_count']} GPUs (DataParallel)")
        else:
            self.gpu_status_label.setText("GPU Status: ✗ Not Available")
            self.gpu_status_label.setStyleSheet("color: orange;")
            self.log("⚠ No GPU detected. Training will use CPU (10-50x slower).")
            self.gpu_radio.setEnabled(False)
            self.cpu_radio.setChecked(True)

    def start_training(self):
        data_dir = self.data_dir_input.text()
        if not data_dir or not os.path.exists(data_dir):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid data directory.")
            return

        model_text = self.model_combo.currentText()
        model_type = model_text.split()[0]

        config = {
            'data_dir': data_dir,
            'output_dir': self.output_dir_input.text(),
            'model_type': model_type,
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'img_size': (self.img_size_spin.value(), self.img_size_spin.value()),
            'num_workers': self.workers_spin.value(),
            'val_split': self.val_split_spin.value(),
            'use_gpu': self.gpu_radio.isChecked()
        }

        os.makedirs(config['output_dir'], exist_ok=True)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.console_output.clear()
        self.epoch_progress.setValue(0)
        self.train_progress.setValue(0)
        self.val_progress.setValue(0)
        
        self.epoch_label.setText(f"Epoch: 0/{config['epochs']}")
        self.train_label.setText("Training: 0/0 batches")
        self.val_label.setText("Validation: 0/0 batches")
        self.total_metrics_label.setText(f"Total - Epochs: {config['epochs']} | Training: Calculating... | Validation: Calculating...")
        
        self.log("="*70)
        self.log("Starting training...")
        self.log(f"Data directory: {config['data_dir']}")
        self.log(f"Model: {config['model_type']}")
        self.log(f"Epochs: {config['epochs']}, Batch size: {config['batch_size']}")
        self.log(f"Device: {'GPU (CUDA)' if config['use_gpu'] else 'CPU'}")
        self.log("="*70)

        self.training_thread = TrainingThread(config)
        self.training_thread.log_signal.connect(self.log)
        self.training_thread.progress_signal.connect(self.update_epoch_progress)
        self.training_thread.training_progress_signal.connect(self.update_training_progress)
        self.training_thread.validation_progress_signal.connect(self.update_validation_progress)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread:
            self.log("Stopping training...")
            self.training_thread.stop()
            self.training_thread.quit()
            self.training_thread.wait()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_epoch_progress(self, current, total):
        progress = int((current / total) * 100)
        self.epoch_progress.setValue(progress)
        self.epoch_label.setText(f"Epoch: {current}/{total}")

    def update_training_progress(self, current, total):
        progress = int((current / total) * 100)
        self.train_progress.setValue(progress)
        self.train_label.setText(f"Training: {current}/{total} batches")
        
        val_total = int(self.val_label.text().split("/")[1].split()[0]) if "/" in self.val_label.text() else 0
        total_epochs = int(self.epoch_label.text().split("/")[1])
        self.total_metrics_label.setText(
            f"Total - Epochs: {total_epochs} | Training: {total} batches/epoch | Validation: {val_total} batches/epoch"
        )

    def update_validation_progress(self, current, total):
        progress = int((current / total) * 100)
        self.val_progress.setValue(progress)
        self.val_label.setText(f"Validation: {current}/{total} batches")
        
        train_total = int(self.train_label.text().split("/")[1].split()[0])
        total_epochs = int(self.epoch_label.text().split("/")[1])
        self.total_metrics_label.setText(
            f"Total - Epochs: {total_epochs} | Training: {train_total} batches/epoch | Validation: {total} batches/epoch"
        )

    def on_training_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.epoch_progress.setValue(100)
            self.train_progress.setValue(100)
            self.val_progress.setValue(100)

        self.log("="*70)
        self.log(message)
        self.log("="*70)

        if success:
            QMessageBox.information(self, "Training Complete", message)
        else:
            QMessageBox.critical(self, "Training Failed", message)

    def log(self, message):
        self.console_output.append(message)
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)


class InferenceTab(QWidget):
    """Inference tab UI with batch inference and single image viewer"""

    def __init__(self):
        super().__init__()
        self.inference_thread = None
        self.prediction_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left side - Configuration and Console
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Configuration section
        config_group = QGroupBox("Inference Configuration")
        config_layout = QVBoxLayout()

        # Model path
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Path:"))
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Select trained model (.pth file)")
        model_layout.addWidget(self.model_path_input)
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn)
        config_layout.addLayout(model_layout)

        # Test directory
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test Directory:"))
        self.test_dir_input = QLineEdit()
        self.test_dir_input.setPlaceholderText("Select folder containing test images")
        test_layout.addWidget(self.test_dir_input)
        self.browse_test_btn = QPushButton("Browse")
        self.browse_test_btn.clicked.connect(self.browse_test_dir)
        test_layout.addWidget(self.browse_test_btn)
        config_layout.addLayout(test_layout)

        # Output path
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output CSV:"))
        self.output_path_input = QLineEdit("submission.csv")
        output_layout.addWidget(self.output_path_input)
        self.browse_csv_btn = QPushButton("Browse")
        self.browse_csv_btn.clicked.connect(self.browse_csv)
        output_layout.addWidget(self.browse_csv_btn)
        config_layout.addLayout(output_layout)

        # Model type
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "unet_resnet50",
            "unet_resnet34",
            "unet_efficientnet_b3",
            "unet_efficientnet_b4"
        ])
        model_type_layout.addWidget(self.model_type_combo)
        config_layout.addLayout(model_type_layout)

        # Parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        params_layout.addWidget(self.batch_spin)

        params_layout.addWidget(QLabel("Image Size:"))
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(128, 1024)
        self.img_size_spin.setValue(512)
        self.img_size_spin.setSingleStep(64)
        params_layout.addWidget(self.img_size_spin)

        params_layout.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(4)
        params_layout.addWidget(self.workers_spin)
        config_layout.addLayout(params_layout)

        # Thresholds
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Class Threshold:"))
        self.class_threshold_spin = QDoubleSpinBox()
        self.class_threshold_spin.setRange(0.0, 1.0)
        self.class_threshold_spin.setValue(0.5)
        self.class_threshold_spin.setSingleStep(0.05)
        threshold_layout.addWidget(self.class_threshold_spin)

        threshold_layout.addWidget(QLabel("Seg Threshold:"))
        self.seg_threshold_spin = QDoubleSpinBox()
        self.seg_threshold_spin.setRange(0.0, 1.0)
        self.seg_threshold_spin.setValue(0.5)
        self.seg_threshold_spin.setSingleStep(0.05)
        threshold_layout.addWidget(self.seg_threshold_spin)
        config_layout.addLayout(threshold_layout)

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # Device selection
        device_group = QGroupBox("Device Selection")
        device_layout = QVBoxLayout()

        device_choice_layout = QHBoxLayout()
        device_choice_layout.addWidget(QLabel("Device:"))

        self.device_button_group = QButtonGroup()

        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.gpu_radio.setChecked(True)
        self.device_button_group.addButton(self.gpu_radio, 0)
        device_choice_layout.addWidget(self.gpu_radio)

        self.cpu_radio = QRadioButton("CPU")
        self.device_button_group.addButton(self.cpu_radio, 1)
        device_choice_layout.addWidget(self.cpu_radio)

        device_choice_layout.addStretch()
        device_layout.addLayout(device_choice_layout)

        status_layout = QHBoxLayout()
        self.gpu_status_label = QLabel("GPU Status: Unknown")
        status_layout.addWidget(self.gpu_status_label)
        status_layout.addStretch()
        device_layout.addLayout(status_layout)

        device_group.setLayout(device_layout)
        left_layout.addWidget(device_group)

        # Control buttons
        control_layout = QHBoxLayout()
        self.run_batch_btn = QPushButton("Run Batch Inference")
        self.run_batch_btn.clicked.connect(self.run_inference)
        self.run_batch_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        control_layout.addWidget(self.run_batch_btn)

        left_layout.addLayout(control_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Console output
        console_group = QGroupBox("Inference Log")
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Consolas", 9))
        console_layout.addWidget(self.console_output)
        console_group.setLayout(console_layout)
        left_layout.addWidget(console_group)

        left_widget.setLayout(left_layout)

        # Right side - Image Viewer
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        viewer_group = QGroupBox("Image Viewer with Predictions")
        viewer_layout = QVBoxLayout()

        # Image selection
        image_select_layout = QHBoxLayout()
        image_select_layout.addWidget(QLabel("Select Image:"))
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("Select an image to view prediction")
        image_select_layout.addWidget(self.image_path_input)
        self.browse_image_btn = QPushButton("Browse")
        self.browse_image_btn.clicked.connect(self.browse_image)
        image_select_layout.addWidget(self.browse_image_btn)
        viewer_layout.addLayout(image_select_layout)

        # Predict button
        self.predict_btn = QPushButton("Predict This Image")
        self.predict_btn.clicked.connect(self.predict_single_image)
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        viewer_layout.addWidget(self.predict_btn)

        # Image display with tabs
        image_tabs = QTabWidget()
        
        # Original image tab
        original_tab = QWidget()
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)
        self.original_image_label.setStyleSheet("border: 1px solid #ccc;")
        original_scroll = QScrollArea()
        original_scroll.setWidget(self.original_image_label)
        original_scroll.setWidgetResizable(True)
        original_layout.addWidget(original_scroll)
        original_tab.setLayout(original_layout)
        image_tabs.addTab(original_tab, "Original")
        
        # Overlay tab
        overlay_tab = QWidget()
        overlay_layout = QVBoxLayout()
        self.overlay_image_label = QLabel()
        self.overlay_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_image_label.setMinimumSize(400, 400)
        self.overlay_image_label.setStyleSheet("border: 1px solid #ccc;")
        overlay_scroll = QScrollArea()
        overlay_scroll.setWidget(self.overlay_image_label)
        overlay_scroll.setWidgetResizable(True)
        overlay_layout.addWidget(overlay_scroll)
        overlay_tab.setLayout(overlay_layout)
        image_tabs.addTab(overlay_tab, "With Annotation")
        
        # Mask tab
        mask_tab = QWidget()
        mask_layout = QVBoxLayout()
        self.mask_image_label = QLabel()
        self.mask_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_image_label.setMinimumSize(400, 400)
        self.mask_image_label.setStyleSheet("border: 1px solid #ccc;")
        mask_scroll = QScrollArea()
        mask_scroll.setWidget(self.mask_image_label)
        mask_scroll.setWidgetResizable(True)
        mask_layout.addWidget(mask_scroll)
        mask_tab.setLayout(mask_layout)
        image_tabs.addTab(mask_tab, "Mask Only")
        
        viewer_layout.addWidget(image_tabs)

        # Metrics display
        metrics_group = QGroupBox("Prediction Metrics")
        metrics_layout = QVBoxLayout()
        
        self.prediction_label = QLabel("Prediction: -")
        self.prediction_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        metrics_layout.addWidget(self.prediction_label)
        
        self.confidence_label = QLabel("Confidence: -")
        metrics_layout.addWidget(self.confidence_label)
        
        self.forged_prob_label = QLabel("Forged Probability: -")
        metrics_layout.addWidget(self.forged_prob_label)
        
        self.authentic_prob_label = QLabel("Authentic Probability: -")
        metrics_layout.addWidget(self.authentic_prob_label)
        
        self.pixels_label = QLabel("Forged Pixels: -")
        metrics_layout.addWidget(self.pixels_label)
        
        self.model_f1_label = QLabel("Model F1 Score: -")
        metrics_layout.addWidget(self.model_f1_label)
        
        self.model_dice_label = QLabel("Model Dice Score: -")
        metrics_layout.addWidget(self.model_dice_label)
        
        metrics_group.setLayout(metrics_layout)
        viewer_layout.addWidget(metrics_group)

        viewer_group.setLayout(viewer_layout)
        right_layout.addWidget(viewer_group)

        right_widget.setLayout(right_layout)

        # Add both sides to splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        QTimer.singleShot(500, self.check_gpu)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pth);;All Files (*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def browse_test_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Test Directory")
        if dir_path:
            self.test_dir_input.setText(dir_path)

    def browse_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Submission File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.output_path_input.setText(file_path)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.image_path_input.setText(file_path)

    def check_gpu(self):
        self.gpu_thread = GPUCheckThread()
        self.gpu_thread.finished.connect(self.on_gpu_check_finished)
        self.gpu_thread.start()

    def on_gpu_check_finished(self, result):
        if result['available']:
            device_info = f"{result['device_count']} GPU(s) - {result['devices'][0]['name']}"
            self.gpu_status_label.setText(f"GPU Status: ✓ {device_info}")
            self.gpu_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.gpu_radio.setEnabled(True)
            self.gpu_radio.setChecked(True)
        else:
            self.gpu_status_label.setText("GPU Status: ✗ Not Available")
            self.gpu_status_label.setStyleSheet("color: orange;")
            self.gpu_radio.setEnabled(False)
            self.cpu_radio.setChecked(True)

    def run_inference(self):
        model_path = self.model_path_input.text()
        test_dir = self.test_dir_input.text()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid model file.")
            return

        if not test_dir or not os.path.exists(test_dir):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid test directory.")
            return

        config = {
            'model_path': model_path,
            'test_dir': test_dir,
            'output_path': self.output_path_input.text(),
            'model_type': self.model_type_combo.currentText(),
            'batch_size': self.batch_spin.value(),
            'img_size': (self.img_size_spin.value(), self.img_size_spin.value()),
            'num_workers': self.workers_spin.value(),
            'class_threshold': self.class_threshold_spin.value(),
            'seg_threshold': self.seg_threshold_spin.value(),
            'use_gpu': self.gpu_radio.isChecked()
        }

        self.run_batch_btn.setEnabled(False)

        self.console_output.clear()
        self.log("="*70)
        self.log("Starting batch inference...")
        self.log(f"Model: {config['model_path']}")
        self.log(f"Test directory: {config['test_dir']}")
        self.log(f"Device: {'GPU (CUDA)' if config['use_gpu'] else 'CPU'}")
        self.log("="*70)

        self.inference_thread = InferenceThread(config)
        self.inference_thread.log_signal.connect(self.log)
        self.inference_thread.progress_signal.connect(self.update_progress)
        self.inference_thread.finished_signal.connect(self.on_inference_finished)
        self.inference_thread.start()

    def predict_single_image(self):
        model_path = self.model_path_input.text()
        image_path = self.image_path_input.text()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid model file.")
            return

        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid image file.")
            return

        self.predict_btn.setEnabled(False)
        self.log(f"Predicting: {os.path.basename(image_path)}")

        device = 'cuda' if self.gpu_radio.isChecked() else 'cpu'
        self.prediction_thread = ImagePredictionThread(
            model_path, image_path, self.model_type_combo.currentText(), device
        )
        self.prediction_thread.finished_signal.connect(self.on_prediction_finished)
        self.prediction_thread.error_signal.connect(self.on_prediction_error)
        self.prediction_thread.start()

    def on_prediction_finished(self, result):
        self.predict_btn.setEnabled(True)

        # Display images
        h, w = result['original_image'].shape[:2]
        
        # Original
        original_rgb = result['original_image']
        original_qimage = QImage(original_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        original_pixmap = QPixmap.fromImage(original_qimage)
        self.original_image_label.setPixmap(original_pixmap.scaled(
            600, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        
        # Overlay
        overlay_rgb = result['overlay']
        overlay_qimage = QImage(overlay_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        overlay_pixmap = QPixmap.fromImage(overlay_qimage)
        self.overlay_image_label.setPixmap(overlay_pixmap.scaled(
            600, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        
        # Mask
        mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
        mask_colored[:, :, 0] = result['mask'] * 255  # Red
        mask_qimage = QImage(mask_colored.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        mask_pixmap = QPixmap.fromImage(mask_qimage)
        self.mask_image_label.setPixmap(mask_pixmap.scaled(
            600, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

        # Display metrics
        prediction_text = "FORGED" if result['is_forged'] else "AUTHENTIC"
        prediction_color = "red" if result['is_forged'] else "green"
        self.prediction_label.setText(f"Prediction: {prediction_text}")
        self.prediction_label.setStyleSheet(f"color: {prediction_color}; font-weight: bold;")
        
        self.confidence_label.setText(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        self.forged_prob_label.setText(f"Forged Probability: {result['forged_prob']:.4f}")
        self.authentic_prob_label.setText(f"Authentic Probability: {result['authentic_prob']:.4f}")
        
        if result['is_forged']:
            percentage = (result['mask_pixels'] / result['total_pixels']) * 100
            self.pixels_label.setText(
                f"Forged Pixels: {result['mask_pixels']:,} / {result['total_pixels']:,} ({percentage:.2f}%)"
            )
        else:
            self.pixels_label.setText("Forged Pixels: 0 (Authentic)")
        
        self.model_f1_label.setText(f"Model F1 Score: {result['val_f1']}")
        self.model_dice_label.setText(f"Model Dice Score: {result['val_dice']}")

        self.log(f"✓ Prediction complete: {prediction_text} (Confidence: {result['confidence']:.4f})")

    def on_prediction_error(self, error_msg):
        self.predict_btn.setEnabled(True)
        self.log(f"✗ {error_msg}")
        QMessageBox.critical(self, "Prediction Failed", error_msg)

    def update_progress(self, current, total):
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)

    def on_inference_finished(self, success, message):
        self.run_batch_btn.setEnabled(True)
        self.progress_bar.setValue(100 if success else 0)

        self.log("="*70)
        self.log(message)
        self.log("="*70)

        if success:
            QMessageBox.information(self, "Inference Complete", message)
        else:
            QMessageBox.critical(self, "Inference Failed", message)

    def log(self, message):
        self.console_output.append(message)
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Scientific Image Forgery Detection - Enhanced GUI")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        title = QLabel("Scientific Image Forgery Detection - With Image Viewer")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("padding: 10px; background-color: #2196F3; color: white;")
        main_layout.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(TrainingTab(), "Training")
        tabs.addTab(InferenceTab(), "Inference & Viewer")

        main_layout.addWidget(tabs)

        self.statusBar().showMessage("Ready | Enhanced with Image Viewer")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
