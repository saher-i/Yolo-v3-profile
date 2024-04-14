import config
import torch
import torch.optim as optim
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, profiler, logger, epoch):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        with profiler.profile("Model Forward and Backward"):
            x = x.to(config.DEVICE)
            y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE))

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                    loss_fn(out[0], y0, scaled_anchors[0]) +
                    loss_fn(out[1], y1, scaled_anchors[1]) +
                    loss_fn(out[2], y2, scaled_anchors[2])
                )

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Update the profiler at the end of each batch
        profiler.step()

        # Update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
    
    # Log average loss of the epoch to TensorBoard
    logger.log_metrics({"epoch_average_loss": mean_loss}, epoch)

def main():
    logger = TensorBoardLogger("tb_logs", name="yolov3_model")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, profiler, logger, epoch)

        # Evaluate model performance on test dataset
        if epoch % 3 == 0:
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD
            )
            map_val = mean_average_precision(
                pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH, box_format="midpoint", num_classes=config.NUM_CLASSES
            )
            print(f"Epoch {epoch} MAP: {map_val.item()}")
            logger.log_metrics({"mAP": map_val.item()}, epoch)

        # Save model periodically
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(model, optimizer, filename=f"checkpoint_{epoch}.pth.tar")

if __name__ == "__main__":
    main()

