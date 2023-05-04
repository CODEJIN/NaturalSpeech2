import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib

class Logger(SummaryWriter):
    def add_scalar_dict(self, scalar_dict, global_step= None, walltime= None):
        for tag, scalar in scalar_dict.items():
            self.add_scalar(
                tag= tag,
                scalar_value= scalar,
                global_step= global_step,
                walltime= walltime
                )
        self.flush()

    def add_image_dict(self, image_dict, global_step, walltime= None):
        for tag, (data, size, aspect, x_limit, y_limit, c_limit) in image_dict.items():
            fig= plt.figure(figsize= size or (10, 5), dpi= 100)
            if data.ndim == 1:
                plt.imshow([[0]], aspect=aspect, origin='lower', cmap= matplotlib.colors.ListedColormap(['white']))
                plt.plot(data)
                plt.margins(x= 0)
                if not x_limit is None:
                    plt.xlim(*x_limit)
                if not y_limit is None:
                    plt.ylim(*y_limit)
            elif data.ndim == 2:
                plt.imshow(data, aspect=aspect, origin='lower')                
                if not x_limit is None:
                    plt.xlim(*x_limit)
                if not y_limit is None:
                    plt.ylim(*y_limit)
                if not c_limit is None:
                    plt.clim(*c_limit)
            elif data.ndim == 3 and data.shape[2] in [3, 4]:    #RGB or RGBA
                plt.imshow(data, aspect=aspect, origin='lower')
                if not x_limit is None:
                    plt.xlim(*x_limit)
                if not y_limit is None:
                    plt.ylim(*y_limit)
                if not c_limit is None:
                    plt.clim(*c_limit)
            plt.colorbar()
            plt.title(tag)
            plt.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            self.add_image(tag= tag, img_tensor= data, global_step= global_step, walltime= walltime, dataformats= 'HWC')
        self.flush()

    def add_audio_dict(self, audio_dict, global_step, walltime= None):
        for tag, (data, sample_rate) in audio_dict.items():
            if data.ndim == 1:
                data = np.expand_dims(data, 0)
            self.add_audio(
                tag= tag,
                snd_tensor= data,
                global_step= global_step,
                sample_rate= sample_rate,
                walltime= walltime
                )
        self.flush()

    def add_histogram_model(self, model, model_label= None, global_step=None, bins='tensorflow', walltime=None, max_bins=None, delete_keywords= []):
        for tag, parameter in model.named_parameters():
            tag = '/'.join([x for x in tag.split('.') if not x in delete_keywords])
            if not model_label is None:
                tag = '{}/{}'.format(model_label, tag)

            self.add_histogram(
                tag= tag,
                values= parameter.data.cpu().numpy(),
                global_step= global_step,
                bins= bins,
                walltime= walltime,
                max_bins= max_bins
                )
            self.flush()