import wandb
import matplotlib.pyplot as plt


def eval_model(model_trainer):
    r"""Performs standard evaluation on the model. Assumes the model has been trained
    and the model's evaluator has been populated
    """
    model_trainer.evaluator.prepare_evaluation_data()
    # Get generic metrics
    evaluator_metrics = model_trainer.evaluator.get_metrics()

    # get line losses plot
    model_trainer.evaluator.plot_line_loss_percentiles(percentiles=[75,95,99], smoothing=10, ylim=(-1,-1), outliers=1, legend=False)
    wandb.log({"Aggregate line losses": plt})
    plt.clf()
    # get roc curve
    _, roc_plot = model_trainer.evaluator.plot_roc_curve(use_wandb=True)
    wandb.log({"ROC Curve": roc_plot})

    # Normalise the data
    model_trainer.evaluator.normalize_losses()

    # get normalised line losses plot
    model_trainer.evaluator.plot_line_loss_percentiles(percentiles=[75,95,99], smoothing=10, ylim=(-1,-1), outliers=1, legend=False)
    wandb.log({"Aggregate line losses (normalised)": plt})
    plt.clf()

    # get normalised roc curve
    evaluator_metrics["eval/AUC_(normalised)"], roc_plot = model_trainer.evaluator.plot_roc_curve(title="ROC (normalised)", use_wandb=True)
    wandb.log({"ROC Curve (normalised)": roc_plot})

    # Log the evaluation results
    #table_columns = [key for key in evaluator_metrics.keys()]
    #table_data = [evaluator_metrics[key] for key in table_columns]
    #metrics_table = wandb.Table(columns=table_columns, data=table_data)
    #wandb.log({"Evaluation metrics": metrics_table})
    for key in evaluator_metrics.keys():
        wandb.run.summary[key] = evaluator_metrics[key]
