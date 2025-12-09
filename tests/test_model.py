#
import pytest
from examples.cifar.src.cnns import model_urls
from examples.cifar.src.utils import load_model


# - SHOULD BE REPLACED BY SOME REGISTRY
def get_image_models():
    unimplemented = ["inception_v3_google"]
    models = list(model_urls.keys())
    models = [m for m in models if m not in unimplemented]
    models += ["vit16b", "vit16l", "vit32l", "vit14h", "vit14g"]

    return models


#
class TestModel:
    """
    Group related tests for 'src/model/' in a class.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        pass

    @pytest.mark.parametrize("model_nme", get_image_models())
    def test_load_model(self, model_nme: str) -> None:
        nb_classes = 10
        load_model(model_nme, nb_classes)
