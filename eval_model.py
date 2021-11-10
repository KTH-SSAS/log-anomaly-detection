import matplotlib.pyplot as plt
import wandb
from log_analyzer.application import Application
from log_analyzer.trainer import Trainer

import wandb


def eval_model(model_trainer: Trainer):
    r"""Performs standard evaluation on the model. Assumes the model has been trained
    and the model's evaluator has been populated
    """
    use_wandb = Application.instance().wandb_initialized
    model_trainer.evaluator.prepare_evaluation_data()
    # Get generic metrics
    evaluator_metrics = model_trainer.evaluator.get_metrics()

    # get line losses plot
    model_trainer.evaluator.plot_line_loss_percentiles(
        percentiles=[75,95,99], smoothing=300, ylim=(-1,-1), outliers=1, legend=False
    )
    if use_wandb:
        wandb.log({"Aggregate line losses": wandb.Image(plt)})
    plt.clf()

    # get roc curve
    _, roc_plot = model_trainer.evaluator.plot_roc_curve(use_wandb=use_wandb)
    if use_wandb:
        wandb.log({"ROC Curve": roc_plot})

    # get pr curve
    AP_score, pr_plot = model_trainer.evaluator.plot_pr_curve(use_wandb=use_wandb)
    if use_wandb:
        wandb.log({"PR Curve": pr_plot})
    evaluator_metrics["eval/AP"] = AP_score


    # Normalise the data
    model_trainer.evaluator.normalize_losses()

    # get normalised line losses plot
    model_trainer.evaluator.plot_line_loss_percentiles(
        percentiles=[75,95,99], smoothing=300, ylim=(-1,-1), outliers=1, legend=False
    )
    if use_wandb:
        wandb.log({"Aggregate line losses (normalised)": wandb.Image(plt)})
    plt.clf()

    # get normalised roc curve
    evaluator_metrics["eval/AUC_(normalised)"], roc_plot = model_trainer.evaluator.plot_roc_curve(
        title="ROC (normalised)", use_wandb=use_wandb
    )
    if use_wandb:
        wandb.log({"ROC Curve (normalised)": roc_plot})

    # get normalised pr curve
    AP_score, pr_plot = model_trainer.evaluator.plot_pr_curve(use_wandb=use_wandb)
    if use_wandb:
        wandb.log({"PR Curve": pr_plot})
    evaluator_metrics["eval/AP_(normalised)"] = AP_score

    # Log the evaluation results
    if use_wandb:
        for key in evaluator_metrics.keys():
            wandb.run.summary[key] = evaluator_metrics[key]
