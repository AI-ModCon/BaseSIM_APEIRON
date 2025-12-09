#
import pytest
import torch
import torch.nn as nn
from examples.cifar.src.cnns import model_urls
from examples.cifar.src.utils import load_model
from training.updaters.jvp_reg import JVPRegularizedLoss


# SHOULD BE REPLACED BY SOMEREGISTRY
def get_image_models():
    unimplemented = ["inception_v3_google"]
    models = list(model_urls.keys())
    models = [m for m in models if m not in unimplemented]
    models += ["vit16b", "vit16l", "vit32l", "vit14h", "vit14g"]
    return models


#
class LogitHarness(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)
        if "logits" in dir(output):
            return output.logits
        else:
            return output[1]


#
class TestTraining:
    """
    Group related test for "src/training/" in a class.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        pass

    @pytest.mark.parametrize("model_nme", get_image_models())
    def test_image_model_forward(self, model_nme: str) -> None:
        # -
        batchsize = 1
        nb_features = 3
        size_image = 518 if model_nme == "vit14g" else 224
        nb_classes = 10

        # -
        model = load_model(model_nme, nb_classes)

        # -
        x_input = torch.zeros((batchsize, nb_features, size_image, size_image))
        output = model(x_input)
        if "logits" in dir(output):
            y_logits = output.logits
        else:
            y_logits = output[1]

        assert list(y_logits.size()) == [batchsize, nb_classes]

    @pytest.mark.parametrize("model_nme", get_image_models())
    def test_updater_jvp_reg(self, model_nme: str) -> None:
        # -
        batchsize = 1
        nb_features = 3
        size_image = 518 if model_nme == "vit14g" else 224
        nb_classes = 10

        # -
        model = load_model(model_nme, nb_classes)
        criterion = nn.CrossEntropyLoss()

        # JVP continual learning setup
        jvp_loss = JVPRegularizedLoss(
            model=LogitHarness(model).train(),
            criterion=criterion,
            # Use defaults for now.
            # jvp_reg=cfg.continuous_learning.jvp_reg,
            # deltax_norm=cfg.continuous_learning.deltax_norm,
        )

        # - Assume dummy data for now
        x_input = torch.zeros((batchsize, nb_features, size_image, size_image))
        x_target = torch.ones((batchsize, nb_classes))
        curr_batch = [x_input, x_target]
        hist_batch = [x_input, x_target]

        jvp_loss(curr_batch, hist_batch)
