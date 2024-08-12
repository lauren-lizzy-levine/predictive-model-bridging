data <-read.table(file.choose(), header=TRUE,stringsAsFactors=TRUE, sep="\t", fill = TRUE, quote = "") # choose file train_dev.tab

t_n_head_deprel <- table(data$t_n_head_deprel, data$bridge)
t_n_head_deprel <- t_n_head_deprel[rowSums(t_n_head_deprel[])>30,]
chisq <- chisq.test(t_n_head_deprel)
residuals <- round(chisq$residuals, 3)
#library(corrplot)
#corrplot(residuals, is.cor = FALSE)

t_n_entity_type <- table(data$t_n_entity_type, data$bridge)
t_n_entity_type <- t_n_entity_type[rowSums(t_n_entity_type[])>30,]
chisq <- chisq.test(t_n_entity_type)
residuals <- round(chisq$residuals, 3)

t_n_head_xpos <- table(data$t_n_head_xpos, data$bridge)
t_n_head_xpos <- t_n_head_xpos[rowSums(t_n_head_xpos[])>30,]
chisq <- chisq.test(t_n_head_xpos)
residuals <- round(chisq$residuals, 3)


n_head_xpos <- table(data$n_head_xpos, data$bridge)
n_head_xpos <- n_head_xpos[rowSums(n_head_xpos[])>10,]
chisq.test(n_head_xpos)
assocplot(n_head_xpos)

t_head_xpos <- table(data$t_head_xpos, data$bridge)
t_head_xpos <- t_head_xpos[rowSums(t_head_xpos[])>10,]
chisq.test(t_head_xpos)
assocplot(t_head_xpos)

genre <- table(data$genre, data$bridge)
chisq.test(genre)
assocplot(genre)

n_entity_type <- table(data$n_entity_type, data$bridge)
n_entity_type <- n_entity_type[rowSums(n_entity_type[])>0,]
chisq.test(n_entity_type)
assocplot(n_entity_type)

t_entity_type <- table(data$t_entity_type, data$bridge)
chisq.test(t_entity_type)
assocplot(t_entity_type)

n_head_deprel <- table(data$n_head_deprel, data$bridge)
n_head_deprel <- n_head_deprel[rowSums(n_head_deprel[])>30,]
chisq.test(n_head_deprel)
assocplot(n_head_deprel)

t_head_deprel <- table(data$t_head_deprel, data$bridge)
t_head_deprel <- t_head_deprel[rowSums(t_head_deprel[])>30,]
chisq.test(t_head_deprel)
assocplot(t_head_deprel)
