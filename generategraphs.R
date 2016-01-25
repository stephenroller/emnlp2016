require(ggplot2)

me <- read.csv("~/working/lexent/matcherror.csv")

ggplot(me[me$features == "diff",], aes(x=recall, y=match_error, color=data, shape=features)) + geom_point(size=1.5) + geom_abline(intercept=0, slope=1) + xlim(0, 1) + ylim(0, 1) + theme_bw() # + scale_color_grey(start = 0.7, end=0.0)
ggplot(me[me$features == "diffsq",], aes(x=recall, y=match_error, color=data, shape=features)) + geom_point(size=1.5) + geom_abline(intercept=0, slope=1) + xlim(0, 1) + ylim(0, 1) + theme_bw() # + scale_color_grey(start = 0.7, end=0.0)
