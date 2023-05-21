# AMP-EF
AMP-EF：A hybrid framework  of machine learning and deep learning for identifying antimicrobial peptides.
In recent years, bacterial resistance has become a serious problem due to the abuse of antibiotics. In this context, the identification and research of Antimicrobial  peptides (AMPs) have become a hot topic in the field of anti-infection. AMPs are amino acid sequences with a broad spectrum of antimicrobial activity. Because of its ability to quickly kill bacteria, fungi, viruses and cancer cells, and offset their toxins, it has quickly become the best alternative to antibiotics . In this study, we constructed a two-branch hybrid framework AMP-EF based on extreme gradient boosting(XGBoost) and bidirectional long short-term memory network(Bi-LSTM) with attention mechanism . As one of the classical machine learning methods, XGBoost has strong stability and can adapt to datasets of different sizes. Bi-LSTM recurse for each amino acid from N-terminal to C-terminal and C-terminal to N-terminal, respectively. As the context information is provided, the model can make more accurate prediction. Our method achieved higher or highly comparable results across eight datasets. The ACC values of XUAMP, YADAMP, DRAMP, CAMP, LAMP, APD3, dbAMP and DBAASP are 77.9%, 98.5%, 72.5%, 99.8%, 83.0%, 92.4%, 87.5% and 84.6%, respectively. This shows that the two-branch structure is feasible and has strong generalization.
