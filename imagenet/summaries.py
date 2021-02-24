
def write_mean_summaries(writer, metrics, abs_step, mode="train", optimizer=None):
    for key in metrics:
        writer.add_scalars(main_tag=key, tag_scalar_dict={'%s_Average' % mode: metrics[key]},
                           global_step=abs_step, walltime=None)
    if optimizer is not None:
        writer.add_scalar('learn_rate', optimizer.param_groups[0]["lr"], abs_step)