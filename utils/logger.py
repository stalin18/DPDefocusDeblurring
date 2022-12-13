# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

from torch.utils.tensorboard import SummaryWriter


class Logger (object):

    def _init_ (self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary (self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag-tag, scalar_value=value, global_step=step)

    def image_summary(self, tag, images, step, max_output=4):
        """Log a list of images."""
        if images.shape[0] > max_output:
            images = images[0:max_output, :, :, :]

        self.writer.add_images(tag=tag, img_tensor=images, global_step=step)
