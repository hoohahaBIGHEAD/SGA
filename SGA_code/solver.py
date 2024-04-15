import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os.path as osp

from model import Generator
from model import Discriminator
from torchvision.models import vgg19
from torchvision.utils import save_image


vgg_activation = dict()


def get_activation(name):
    """
    Returns a hook function that saves the output of a layer's activation.

    Parameters:
    name (str): The name of the layer.

    Returns:
    hook (function): A hook function that saves the output of the layer's activation.

    """

    def hook(model, input, output):
        """
        This function is a hook that is called when a layer in the model is executed.
        It stores the output of the layer in the `vgg_activation` dictionary.

        Parameters:
        - model: The model being executed.
        - input: The input to the layer.
        - output: The output of the layer.

        Returns:
        None
        """
        vgg_activation[name] = output

    return hook


def gram_matrix(x):
    """
    Calculate the Gram matrix of the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (a, b, c, d).

    Returns:
        torch.Tensor: Gram matrix of shape (a * b, a * b).

    """
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class Solver(object):
    """
    The Solver class is responsible for training the model and managing the training process.

    Args:
        config (dict): Configuration parameters for the model.
        data_loader (DataLoader): The data loader object that provides the training data.

    Attributes:
        data_loader (DataLoader): The data loader object that provides the training data.
        img_size (int): The size of the input images.
        epoch (int): The number of training epochs.
        batch_size (int): The batch size for training.
        g_lr (float): The learning rate for the generator.
        d_lr (float): The learning rate for the discriminator.
        lambda_g_fake (float): The weight for the fake loss in the generator.
        lambda_g_recon (float): The weight for the reconstruction loss in the generator.
        lambda_g_style (float): The weight for the style loss in the generator.
        lambda_g_percep (float): The weight for the perceptual loss in the generator.
        lambda_d_fake (float): The weight for the fake loss in the discriminator.
        lambda_d_real (float): The weight for the real loss in the discriminator.
        d_critic (int): The number of times the discriminator is updated per generator update.
        g_critic (int): The number of times the generator is updated per discriminator update.
        mse_loss (nn.MSELoss): The mean squared error loss function.
        optim (str): The optimizer used for training.
        beta1 (float): The beta1 parameter for the Adam optimizer.
        beta2 (float): The beta2 parameter for the Adam optimizer.
        adversarial_loss (nn.MSELoss): The adversarial loss function.
        l1_loss (nn.L1Loss): The L1 loss function.
        cpu_seed (int): The random seed for CPU operations.
        gpu_seed (int): The random seed for GPU operations.
        g_spec (bool): Whether to use spectral normalization in the generator.
        d_spec (bool): Whether to use spectral normalization in the discriminator.
        gpu (int): The index of the GPU device to use for training.
        use_tensorboard (bool): Whether to use TensorBoard for logging.
        target_layer (list): The target layers for feature extraction.
        train_dir (str): The directory for saving training-related files.
        log_dir (str): The directory for saving log files.
        sample_dir (str): The directory for saving sample images.
        result_dir (str): The directory for saving result images.
        model_dir (str): The directory for saving model checkpoints.
        log_step (int): The interval for logging training progress.
        sample_step (int): The interval for saving sample images.
        save_step (int): The interval for saving model checkpoints.
        save_start (int): The epoch to start saving model checkpoints.
        lr_decay_step (int): The interval for decaying the learning rate.
        vgg (nn.Module): The VGG19 model for perceptual loss calculation.
        D (Discriminator): The discriminator model.
        G (Generator): The generator model.
        g_optimizer (torch.optim.Adam): The optimizer for the generator.
        d_optimizer (torch.optim.Adam): The optimizer for the discriminator.

    Methods:
        build_model(): Builds the VGG, discriminator, and generator models.
        print_network(model, name): Prints the network information for a given model.
        update_lr(g_lr, d_lr): Updates the learning rates of the generator and discriminator.
        reset_grad(): Resets the gradient buffers.
        denorm(x): Converts the range of values from [-1, 1] to [0, 1].
        restore_model(): Restores the trained model from the latest checkpoint.
        image_reporting(fixed_sketch, fixed_reference, fixed_elastic_reference, epoch, postfix=""): Generates and saves sample images during training.
        percep_style(reference, fake_images): Calculates the perceptual and style losses between the reference and fake images.
        G_train(sketch, reference, loss_dict, elastic_reference): Performs a forward pass and updates the generator.
        D_train(sketch, reference, loss_dict, elastic_reference): Performs a forward pass and updates the discriminator.
        train(): Trains the model using the provided data loader.
    """

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.data_loader = data_loader
        self.img_size = config["MODEL_CONFIG"]["IMG_SIZE"]
        assert self.img_size in [256]

        self.epoch = config["TRAINING_CONFIG"]["EPOCH"]
        self.batch_size = config["TRAINING_CONFIG"]["BATCH_SIZE"]
        self.g_lr = float(config["TRAINING_CONFIG"]["G_LR"])
        self.d_lr = float(config["TRAINING_CONFIG"]["D_LR"])
        self.lambda_g_fake = config["TRAINING_CONFIG"]["LAMBDA_G_FAKE"]
        self.lambda_g_recon = config["TRAINING_CONFIG"]["LAMBDA_G_RECON"]
        self.lambda_g_style = config["TRAINING_CONFIG"]["LAMBDA_G_SYTLE"]
        self.lambda_g_percep = config["TRAINING_CONFIG"]["LAMBDA_G_PERCEP"]
        self.lambda_d_fake = config["TRAINING_CONFIG"]["LAMBDA_D_FAKE"]
        self.lambda_d_real = config["TRAINING_CONFIG"]["LAMBDA_D_REAL"]
        self.d_critic = config["TRAINING_CONFIG"]["D_CRITIC"]
        self.g_critic = config["TRAINING_CONFIG"]["G_CRITIC"]
        self.mse_loss = nn.MSELoss()

        self.optim = config["TRAINING_CONFIG"]["OPTIM"]
        self.beta1 = config["TRAINING_CONFIG"]["BETA1"]
        self.beta2 = config["TRAINING_CONFIG"]["BETA2"]

        self.adversarial_loss = torch.nn.MSELoss()

        self.l1_loss = torch.nn.L1Loss()

        self.cpu_seed = config["TRAINING_CONFIG"]["CPU_SEED"]
        self.gpu_seed = config["TRAINING_CONFIG"]["GPU_SEED"]

        self.g_spec = config["TRAINING_CONFIG"]["G_SPEC"] == "True"
        self.d_spec = config["TRAINING_CONFIG"]["D_SPEC"] == "True"

        self.gpu = config["TRAINING_CONFIG"]["GPU"]
        self.use_tensorboard = config["TRAINING_CONFIG"]["USE_TENSORBOARD"]

        self.target_layer = ["relu_3", "relu_8"]

        # Directory
        self.train_dir = config["TRAINING_CONFIG"]["TRAIN_DIR"]
        self.log_dir = os.path.join(
            self.train_dir, config["TRAINING_CONFIG"]["LOG_DIR"]
        )
        self.sample_dir = os.path.join(
            self.train_dir, config["TRAINING_CONFIG"]["SAMPLE_DIR"]
        )
        self.result_dir = os.path.join(
            self.train_dir, config["TRAINING_CONFIG"]["RESULT_DIR"]
        )
        self.model_dir = os.path.join(
            self.train_dir, config["TRAINING_CONFIG"]["MODEL_DIR"]
        )

        # Steps
        self.log_step = config["TRAINING_CONFIG"]["LOG_STEP"]
        self.sample_step = config["TRAINING_CONFIG"]["SAMPLE_STEP"]
        self.save_step = config["TRAINING_CONFIG"]["SAVE_STEP"]
        self.save_start = config["TRAINING_CONFIG"]["SAVE_START"]
        self.lr_decay_step = config["TRAINING_CONFIG"]["LR_DECAY_STEP"]

        # build net
        self.vgg = vgg19(pretrained=True)

        self.D = Discriminator(spec_norm=self.d_spec, LR=0.2).to(self.gpu)
        self.G = Generator(spec_norm=self.g_spec).to(self.gpu)

        # optimizer
        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.G.parameters()),
            self.g_lr,
            (self.beta1, self.beta2),
        )

        self.d_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D.parameters()),
            self.d_lr,
            (self.beta1, self.beta2),
        )
        self.build_model()

    def build_model(self):
        """
        Builds the model by registering forward hooks for target layers,
        moving the model to the GPU, and printing the network architecture.

        Returns:
            None
        """
        for layer in self.target_layer:
            self.vgg.features[int(layer.split("_")[-1])].register_forward_hook(
                get_activation(layer)
            )
        self.vgg.to(self.gpu)
        self.print_network(self.G, "G")
        self.print_network(self.D, "D")

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir, "model_arch.txt"), "a") as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params), file=fp)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    @staticmethod
    def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def restore_model(self):
        """
        Restores the model from the latest saved checkpoint.

        Returns:
            int: The epoch number of the restored model.
        """
        pth_list = glob.glob(osp.join(self.model_dir, "*-G.pth"))

        if len(pth_list) == 0:
            return 0

        pth_list = [int(osp.basename(x).split("-")[0]) for x in pth_list]
        pth_list.sort()
        epoch = pth_list[-1]
        G_path = os.path.join(self.model_dir, "{}-G.pth".format(epoch))
        D_path = os.path.join(self.model_dir, "{}-D.pth".format(epoch))

        G_checkpoint = torch.load(G_path)
        self.G.load_state_dict(G_checkpoint["model"])
        self.g_optimizer.load_state_dict(G_checkpoint["optimizer"])

        D_checkpoint = torch.load(D_path)
        self.D.load_state_dict(D_checkpoint["model"])
        self.d_optimizer.load_state_dict(D_checkpoint["optimizer"])

        self.G.to(self.gpu)
        self.D.to(self.gpu)
        return epoch

    def image_reporting(
        self, fixed_sketch, fixed_reference, fixed_elastic_reference, epoch, postfix=""
    ):
        """
        Generate and save image reports for the given inputs.

        Args:
            fixed_sketch (torch.Tensor): The fixed sketch input.
                The input tensor representing the fixed sketch.
            fixed_reference (torch.Tensor): The fixed reference input.
                The input tensor representing the fixed reference.
            fixed_elastic_reference (torch.Tensor): The fixed elastic reference input.
                The input tensor representing the fixed elastic reference.
            epoch (int): The current epoch number.
                The number of the current epoch.
            postfix (str, optional): A postfix string to append to the saved image file name. Defaults to "".
                An optional string to append to the saved image file name.

        Returns:
            None
        """
        image_report = list()
        image_report.append(fixed_sketch.expand_as(fixed_reference))
        image_report.append(fixed_elastic_reference)
        image_report.append(fixed_reference)

        fake_result = self.G(fixed_elastic_reference, fixed_sketch)
        image_report.append(fake_result)

        x_concat = torch.cat(image_report, dim=3)
        sample_path = os.path.join(
            self.sample_dir, "{}-images{}.jpg".format(epoch, postfix)
        )
        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

    def percep_style(self, reference, fake_images):
        """
        Calculates the perceptual loss for style and perception between the reference image and the fake images.

        Args:
            reference (Tensor): The reference image.
            fake_images (Tensor): The generated fake images.

        Returns:
            Tuple: A tuple containing the perceptual loss for perception and style.

        """
        fake_activation = dict()
        real_activation = dict()

        # percep_style
        g_loss_style = 0
        g_loss_percep = 0

        self.vgg(reference)
        for layer in self.target_layer:
            real_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        self.vgg(fake_images)
        for layer in self.target_layer:
            fake_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        for layer in self.target_layer:
            g_loss_percep += self.l1_loss(
                fake_activation[layer], real_activation[layer]
            )
            g_loss_style += self.l1_loss(
                gram_matrix(fake_activation[layer]), gram_matrix(real_activation[layer])
            )

        return g_loss_percep, g_loss_style

    def G_train(self, sketch, reference, loss_dict, elastic_reference):
        """
        Trains the generator network.

        Args:
            sketch (torch.Tensor): The input sketch tensor.
            reference (torch.Tensor): The reference image tensor.
            loss_dict (dict): A dictionary to store the loss values.
            elastic_reference (torch.Tensor): The elastic reference tensor.

        Returns:
            None
        """
        fake_images = self.G(elastic_reference, sketch)

        fake_score = self.D(torch.cat([fake_images, sketch], dim=1))

        g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))

        g_loss_recon = self.l1_loss(fake_images, reference)

        g_loss_percep, g_loss_style = self.percep_style(reference, fake_images)

        g_loss = (
            self.lambda_g_fake * g_loss_fake
            + self.lambda_g_percep * g_loss_percep
            + self.lambda_g_style * g_loss_style
            + self.lambda_g_recon * g_loss_recon
        )

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # Logging.
        loss_dict["G/loss_fake"] = self.lambda_g_fake * g_loss_fake.item()
        loss_dict["G/loss_style"] = self.lambda_g_style * g_loss_style.item()
        loss_dict["G/loss_percep"] = self.lambda_g_percep * g_loss_percep.item()
        loss_dict["G/loss_recon"] = self.lambda_g_recon * g_loss_recon.item()

    def D_train(self, sketch, reference, loss_dict, elastic_reference):
        """
        Trains the discriminator network.

        Args:
            sketch (torch.Tensor): The input sketch tensor.
            reference (torch.Tensor): The reference image tensor.
            loss_dict (dict): A dictionary to store the loss values.
            elastic_reference (torch.Tensor): The elastic reference image tensor.

        Returns:
            None
        """
        fake_images = self.G(elastic_reference, sketch)

        real_score = self.D(torch.cat([reference, sketch], dim=1))
        fake_score = self.D(torch.cat([fake_images.detach(), sketch], dim=1))

        d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
        d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))

        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake

        # Backward and optimize.
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        loss_dict["D/loss_real"] = self.lambda_d_real * d_loss_real.item()
        loss_dict["D/loss_fake"] = self.lambda_d_fake * d_loss_fake.item()

    def train(self):
        """
        Trains the model using the specified data loader and hyperparameters.

        Returns:
            None
        """
        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print("iterations : ", iterations)

        data_iter = iter(data_loader)
        _, fixed_elastic_reference, fixed_reference, fixed_sketch = next(data_iter)

        split_fixed_sketch = list(torch.chunk(fixed_sketch, self.batch_size, dim=0))
        first_fixed_sketch = split_fixed_sketch[0]
        del split_fixed_sketch[0]
        split_fixed_sketch.append(first_fixed_sketch)
        shifted_fixed_sketch = torch.cat(split_fixed_sketch, dim=0)

        fixed_sketch = fixed_sketch.to(self.gpu)
        fixed_reference = fixed_reference.to(self.gpu)
        fixed_elastic_reference = fixed_elastic_reference.to(self.gpu)
        shifted_fixed_sketch = shifted_fixed_sketch.to(self.gpu)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_epoch = self.restore_model()
        start_time = time.time()
        print("Start training...")
        for e in range(start_epoch, self.epoch):

            for i in range(iterations):
                try:
                    _, elastic_reference, reference, sketch = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, elastic_reference, reference, sketch = next(data_iter)

                elastic_reference = elastic_reference.to(self.gpu)
                reference = reference.to(self.gpu)
                sketch = sketch.to(self.gpu)

                loss_dict = dict()
                if (i + 1) % self.d_critic == 0:
                    self.D_train(sketch, reference, loss_dict, elastic_reference)
                if (i + 1) % self.g_critic == 0:
                    self.G_train(sketch, reference, loss_dict, elastic_reference)

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(
                        e + 1, self.epoch, et, i + 1, iterations
                    )
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    txt_name = f"{e + 1}.txt"
                    with open(os.path.join(self.log_dir, txt_name), "a") as fp:
                        print(log, file=fp)

            if (e + 1) % self.sample_step == 0:
                self.G.eval()
                with torch.no_grad():
                    self.image_reporting(
                        fixed_sketch,
                        fixed_reference,
                        fixed_elastic_reference,
                        e + 1,
                        postfix="",
                    )
                    self.image_reporting(
                        shifted_fixed_sketch,
                        fixed_reference,
                        fixed_elastic_reference,
                        e + 1,
                        postfix="_shifted",
                    )
                    print(
                        "Saved real and fake images into {}...".format(self.sample_dir)
                    )
                self.G.train()

                # Save model checkpoints.
                if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                    G_path = os.path.join(self.model_dir, "{}-G.pth".format(e + 1))
                    D_path = os.path.join(self.model_dir, "{}-D.pth".format(e + 1))

                    G_state = {
                        "model": self.G.state_dict(),
                        "optimizer": self.g_optimizer.state_dict(),
                    }
                    D_state = {
                        "model": self.D.state_dict(),
                        "optimizer": self.d_optimizer.state_dict(),
                    }

                    torch.save(G_state, G_path)
                    torch.save(D_state, D_path)
                    print("Saved model checkpoints into {}...".format(self.model_dir))

        print("Training is finished")

    def test(self):
        pass
