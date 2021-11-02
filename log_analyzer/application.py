"""Application level singleton (fancy term for global) variables."""

TRAINER_LOGGER = "trainer-logger"


class Application:

    __instance = None

    artifact_name = ""  # Name to use for saving model to W&B

    @classmethod
    def instance(cls):
        """For smooth integration, the application will init itself with
        default options if the instance is requested before being explicitly
        initalized."""
        if Application.__instance is None:
            Application.__instance = Application()

        return Application.__instance

    def __init__(self, cuda=False, wandb=False) -> None:
        if Application.__instance is not None:
            raise Exception("Application has already been initialized.")

        Application.__instance = self

        self._use_cuda = cuda
        self._wandb_initalized = wandb

    @property
    def using_cuda(self):
        return self._use_cuda

    @property
    def wandb_initialized(self):
        return self._wandb_initalized
