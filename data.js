const categories = {
    "Importing Data": [
        "CSV", "Excel", "PDF", "Word"
    ],
    "Data Exploration & Summary": [
        "Value Counts & Unique Values",
        "Descriptive Statistics",
        "Frequency Tables",
        "Missing Value Summary",
        "Data Types Overview",
        "Outlier Detection"
    ],
    "Data Visualization": [
        "Histogram",
        "Boxplot",
        "Bar Plot",
        "Pie Chart",
        "Scatter Plot",
        "Line Plot",
        "Heatmap",
        "Pairplot",
        "Violin Plot",
        "Correlation Heat Map"
    ],
    "Data Cleaning & Preprocessing": [
        "Handle Missing Values",
        "Remove Duplicates",
        "Encode Categorical Variables",
        "Normalize/Standardize Data",
        "Outlier Removal",
        "Feature Scaling",
        "Data Splitting",
        "Data Type Conversion"
    ],
    "Statistical Tests": [
        "One-Sample T-Test",
        "Independent Two-Sample T-Test",
        "Paired Two-Sample T-Test",
        "ANOVA",
        "Chi-Square Test",
        "Mann-Whitney U Test",
        "Wilcoxon Test",
        "Kruskal-Wallis Test",
        "Normality Tests",
        "Levene’s Test"
    ],
    "Correlation & Association": [
        "Pearson Correlation Matrix",
        "Spearman Correlation Matrix",
        "Partial Correlations",
        "Crosstab / Contingency Table",
        "Covariance Matrix",
        "Association Rules"
    ],
    "Regression Analysis": [
        "Linear Regression",
        "Multiple Regression",
        "Logistic Regression",
        "Polynomial Regression",
        "Ridge/Lasso Regression",
        "Poisson Regression"
    ],
    "Machine Learning": [
        "Train/Test Split",
        "Cross Validation",
        "Classification (SVM, Decision Tree, Random Forest, KNN)",
        "Regression (Random Forest, SVR, etc.)",
        "Clustering (KMeans, Hierarchical, DBSCAN)",
        "K-Means Clustering",
        "Principal Component Analysis",
        "Model Evaluation",
        "Feature Importance"
    ],
    "Time Series Analysis": [
        "Time Series Decomposition",
        "Autocorrelation Plot",
        "ARIMA Modeling",
        "Forecasting"
    ],
    "Dimensionality Reduction": [
        "PCA",
        "t-SNE",
        "UMAP"
    ],
    "Data Export & Reporting": [
        "Export DataFrame to CSV/Excel",
        "Generate Summary Report"
    ]
};

const codeTemplates = {
    "CSV": {
        "Python": `import pandas as pd\ndf = pd.read_csv('file.csv')`,
        "R": `df <- read.csv('file.csv')`
    },
    "Excel": {
        "Python": `import pandas as pd\ndf = pd.read_excel('file.xlsx')`,
        "R": `library(readxl)\ndf <- read_excel('file.xlsx')`
    },
    "PDF": {
        "Python": `import tabula\ndf = tabula.read_pdf('file.pdf', pages='all')[0]`,
        "R": `library(tabulizer)\ndf <- extract_tables('file.pdf')`
    },
    "Word": {
        "Python": `from docx import Document\ndoc = Document('file.docx')\ntext = '\n'.join([para.text for para in doc.paragraphs])`,
        "R": `library(officer)\ndoc <- read_docx('file.docx')\n# Extract text from Word in R is more complex; see officer documentation.`
    },
    "Value Counts & Unique Values": {
        "Python": `value_counts = df['your_column'].value_counts()\nunique_values = df['your_column'].unique()\nprint(value_counts)\nprint(unique_values)`,
        "R": `value_counts <- table(df$your_column)\nunique_values <- unique(df$your_column)\nprint(value_counts)\nprint(unique_values)`
    },
    "Descriptive Statistics": {
        "Python": `# Get a full summary of descriptive statistics
print(df.describe())

# Get individual statistics
mean_val = df['your_column'].mean()
median_val = df['your_column'].median()
mode_val = df['your_column'].mode()[0]
std_dev = df['your_column'].std()
min_val = df['your_column'].min()
max_val = df['your_column'].max()

print(f"\nMean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
print(f"Standard Deviation: {std_dev}")
print(f"Minimum: {min_val}")
print(f"Maximum: {max_val}")`,
        "R": `# Get a full summary of descriptive statistics
summary(df)

# Get individual statistics
mean_val <- mean(df$your_column, na.rm = TRUE)
median_val <- median(df$your_column, na.rm = TRUE)
std_dev <- sd(df$your_column, na.rm = TRUE)
min_val <- min(df$your_column, na.rm = TRUE)
max_val <- max(df$your_column, na.rm = TRUE)

# For mode, R doesn't have a built-in function, but you can calculate it
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
mode_val <- get_mode(df$your_column)

print(paste("Mean:", mean_val))
print(paste("Median:", median_val))
print(paste("Mode:", mode_val))
print(paste("Standard Deviation:", std_dev))
print(paste("Minimum:", min_val))
print(paste("Maximum:", max_val))`
    },
    "Frequency Tables": {
        "Python": `freq_table = pd.crosstab(index=df['your_column'], columns="count")\nprint(freq_table)`,
        "R": `freq_table <- table(df$your_column)\nprint(freq_table)`
    },
    "Missing Value Summary": {
        "Python": `missing = df.isnull().sum()\nprint(missing)`,
        "R": `missing <- colSums(is.na(df))\nprint(missing)`
    },
    "Data Types Overview": {
        "Python": `print(df.dtypes)`,
        "R": `sapply(df, class)`
    },
    "Outlier Detection": {
        "Python": `Q1 = df['your_column'].quantile(0.25)\nQ3 = df['your_column'].quantile(0.75)\nIQR = Q3 - Q1\noutliers = df[(df['your_column'] < (Q1 - 1.5 * IQR)) | (df['your_column'] > (Q3 + 1.5 * IQR))]\nprint(outliers)`,
        "R": `Q1 <- quantile(df$your_column, 0.25)\nQ3 <- quantile(df$your_column, 0.75)\nIQR <- Q3 - Q1\noutliers <- df[df$your_column < (Q1 - 1.5 * IQR) | df$your_column > (Q3 + 1.5 * IQR), ]\nprint(outliers)`
    },
    "Histogram": {
        "Python": `import matplotlib.pyplot as plt\ndf['your_column'].hist()\nplt.xlabel('your_column')\nplt.ylabel('Frequency')\nplt.show()`,
        "R": `hist(df$your_column, main="Histogram", xlab="your_column", col="skyblue")`
    },
    "Boxplot": {
        "Python": `import matplotlib.pyplot as plt\ndf.boxplot(column='your_column')\nplt.show()`,
        "R": `boxplot(df$your_column, main="Boxplot")`
    },
    "Bar Plot": {
        "Python": `import matplotlib.pyplot as plt\ndf['your_column'].value_counts().plot(kind='bar')\nplt.show()`,
        "R": `barplot(table(df$your_column), main="Bar Plot")`
    },
    "Pie Chart": {
        "Python": `import matplotlib.pyplot as plt\ndf['your_column'].value_counts().plot.pie(autopct='%1.1f%%')\nplt.ylabel('')\nplt.show()`,
        "R": `pie(table(df$your_column), main="Pie Chart")`
    },
    "Scatter Plot": {
        "Python": `import matplotlib.pyplot as plt\nplt.scatter(df['x'], df['y'])\nplt.xlabel('x')\nplt.ylabel('y')\nplt.show()`,
        "R": `plot(df$x, df$y, main="Scatter Plot", xlab="x", ylab="y")`
    },
    "Line Plot": {
        "Python": `import matplotlib.pyplot as plt\nplt.plot(df['x'], df['y'])\nplt.xlabel('x')\nplt.ylabel('y')\nplt.show()`,
        "R": `plot(df$x, df$y, type="l", main="Line Plot", xlab="x", ylab="y")`
    },
    "Heatmap": {
        "Python": `import seaborn as sns\nimport matplotlib.pyplot as plt\nsns.heatmap(df.corr(), annot=True)\nplt.show()`,
        "R": `library(gplots)\nheatmap.2(as.matrix(cor(df)), trace="none")`
    },
    "Pairplot": {
        "Python": `import seaborn as sns\nsns.pairplot(df)\nplt.show()`,
        "R": `pairs(df)`
    },
    "Violin Plot": {
        "Python": `import seaborn as sns\nsns.violinplot(x=df['your_column'])\nplt.show()`,
        "R": `library(vioplot)\nvioplot(df$your_column)`
    },
    "Correlation Heat Map": {
        "Python": `import seaborn as sns\nimport matplotlib.pyplot as plt\n\n# Ensure you have a dataframe 'df' with numerical columns\ncorr = df.corr()\nplt.figure(figsize=(10, 8))\nsns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Correlation Heatmap')\nplt.show()`,
        "R": `library(corrplot)\nlibrary(RColorBrewer)\n\n# Ensure you have a dataframe 'df' with numerical columns\ncorr <- cor(df)\ncorrplot(corr, method="color", col=brewer.pal(n=8, name="RdYlBu"),\n         type="upper", order="hclust",\n         addCoef.col = "black", \n         tl.col="black", tl.srt=45, \n         diag=FALSE\n)`
    },
    "Data Type Conversion": {
        "Python": `# Convert column to numeric\n# Use pd.to_numeric() to convert a column to numeric type\ndf['your_column'] = pd.to_numeric(df['your_column'], errors='coerce')\n\n# Convert column to datetime\n# Use pd.to_datetime() to convert a column to datetime type\ndf['your_column'] = pd.to_datetime(df['your_column'])\n\n# Convert column to string\n# Use astype() to convert a column to string type\ndf['your_column'] = df['your_column'].astype(str)`,
        "R": `# Convert column to numeric\n# Use as.numeric() to convert a column to numeric type\ndf$your_column <- as.numeric(as.character(df$your_column))\n# Convert column to date\n# Use as.Date() to convert a column to date type\ndf$your_column <- as.Date(df$your_column)\n# Convert column to character\n# Use as.character() to convert a column to character type\ndf$your_column <- as.character(df$your_column)`
    },
    "Handle Missing Values": {
        "Python": `# Drop missing values\ndf_clean = df.dropna()\n# Fill missing values\ndf_filled = df.fillna(0)`,
        "R": `# Drop missing values\ndf_clean <- na.omit(df)\n# Fill missing values\ndf_filled <- df\ndf_filled[is.na(df_filled)] <- 0`
    },
    "Remove Duplicates": {
        "Python": `df_cleaned = df.drop_duplicates()`,
        "R": `df_cleaned <- unique(df)`
    },
    "Encode Categorical Variables": {
        "Python": `# One-hot encoding\ndf_encoded = pd.get_dummies(df, columns=['your_column'])`,
        "R": `# Factor encoding\ndf$your_column <- as.factor(df$your_column)`
    },
    "Normalize/Standardize Data": {
        "Python": `from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndf_scaled = scaler.fit_transform(df)`,
        "R": `df_scaled <- scale(df)`
    },
    "Outlier Removal": {
        "Python": `Q1 = df['col'].quantile(0.25)\nQ3 = df['col'].quantile(0.75)\nIQR = Q3 - Q1\ndf_no_outliers = df[~((df['col'] < (Q1 - 1.5 * IQR)) | (df['col'] > (Q3 + 1.5 * IQR)))]`,
        "R": `Q1 <- quantile(df$col, 0.25)\nQ3 <- quantile(df$col, 0.75)\nIQR <- Q3 - Q1\ndf_no_outliers <- subset(df, df$col >= (Q1 - 1.5*IQR) & df$col <= (Q3 + 1.5*IQR))`
    },
    "Feature Scaling": {
        "Python": `from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\ndf_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)`,
        "R": `df_scaled <- as.data.frame(scale(df, center=FALSE, scale=apply(df,2,max)))`
    },
    "Data Splitting": {
        "Python": `from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`,
        "R": `set.seed(123)\ntrain_idx <- sample(seq_len(nrow(df)), size = 0.8*nrow(df))\ntrain <- df[train_idx, ]\ntest <- df[-train_idx, ]`
    },
    "One-Sample T-Test": {
        "Python": `from scipy import stats\n# Test if the mean of a sample is equal to a given value\nt_stat, p_val = stats.ttest_1samp(df['your_column'], popmean=50)\nprint(f"t-statistic: {t_stat}, p-value: {p_val}")`,
        "R": `# Test if the mean of a sample is equal to a given value\ntest_result <- t.test(df$your_column, mu = 50)\nprint(test_result)`
    },
    "Independent Two-Sample T-Test": {
        "Python": `from scipy import stats\n# Independent t-test\nt_stat, p_val = stats.ttest_ind(df['group1'], df['group2'])\nprint(f"t-statistic: {t_stat}, p-value: {p_val}")`,
        "R": `# Independent t-test\ntest_result <- t.test(df$group1, df$group2)\nprint(test_result)`
    },
    "Paired Two-Sample T-Test": {
        "Python": `from scipy import stats\n# Paired t-test\nt_stat, p_val = stats.ttest_rel(df['group1_paired'], df['group2_paired'])\nprint(f"t-statistic: {t_stat}, p-value: {p_val}")`,
        "R": `# Paired t-test\ntest_result <- t.test(df$group1_paired, df$group2_paired, paired = TRUE)\nprint(test_result)`
    },
    "ANOVA": {
        "Python": `from scipy.stats import f_oneway\nf_stat, p_val = f_oneway(df['group1'], df['group2'], df['group3'])\nprint(f_stat, p_val)`,
        "R": `res.aov <- aov(value ~ group, data = df)\nsummary(res.aov)`
    },
    "Chi-Square Test": {
        "Python": `from scipy.stats import chi2_contingency\nchi2, p, dof, expected = chi2_contingency(pd.crosstab(df['var1'], df['var2']))\nprint(p)`,
        "R": `chisq.test(table(df$var1, df$var2))`
    },
    "Mann-Whitney U Test": {
        "Python": `from scipy.stats import mannwhitneyu\nstat, p = mannwhitneyu(df['group1'], df['group2'])\nprint(p)`,
        "R": `wilcox.test(value ~ group, data=df)`
    },
    "Wilcoxon Test": {
        "Python": `from scipy.stats import wilcoxon\nstat, p = wilcoxon(df['group1'], df['group2'])\nprint(p)`,
        "R": `wilcox.test(df$group1, df$group2, paired=TRUE)`
    },
    "Kruskal-Wallis Test": {
        "Python": `from scipy.stats import kruskal\nstat, p = kruskal(df['group1'], df['group2'], df['group3'])\nprint(p)`,
        "R": `kruskal.test(value ~ group, data=df)`
    },
    "Normality Tests": {
        "Python": `from scipy import stats\n# Shapiro-Wilk Test\nshapiro_stat, shapiro_p = stats.shapiro(df['your_column'])\nprint(f"Shapiro-Wilk: Statistics={shapiro_stat}, p-value={shapiro_p}")\n\n# Kolmogorov-Smirnov test for normality\nks_stat, ks_p = stats.kstest(df['your_column'], 'norm')\nprint(f"Kolmogorov-Smirnov: Statistics={ks_stat}, p-value={ks_p}")`,
        "R": `# Shapiro-Wilk Test\nshapiro.test(df$your_column)\n\n# Kolmogorov-Smirnov test for normality\nks.test(df$your_column, "pnorm")`
    },
    "Levene’s Test": {
        "Python": `from scipy.stats import levene\nstat, p = levene(df['group1'], df['group2'])\nprint(stat, p)`,
        "R": `library(car)\nleveneTest(value ~ group, data=df)`
    },
    "Pearson Correlation Matrix": {
        "Python": `# Pearson is the default method\ncorr_matrix = df.corr(method='pearson')\nprint(corr_matrix)`,
        "R": `# Pearson is the default method\ncorr_matrix <- cor(df, method = "pearson")\nprint(corr_matrix)`
    },
    "Spearman Correlation Matrix": {
        "Python": `# Spearman correlation for non-parametric data\ncorr_matrix = df.corr(method='spearman')\nprint(corr_matrix)`,
        "R": `# Spearman correlation for non-parametric data\ncorr_matrix <- cor(df, method = "spearman")\nprint(corr_matrix)`
    },
    "Partial Correlations": {
        "Python": `# You may need to install pingouin: pip install pingouin\nimport pingouin as pg\n# Partial correlation of x and y, controlling for z\npartial_corr = pg.partial_corr(data=df, x='x_col', y='y_col', covar='z_col')\nprint(partial_corr)`,
        "R": `# You may need to install ppcor: install.packages("ppcor")\nlibrary(ppcor)\n# Partial correlation between two variables, controlling for a third\npartial_corr <- pcor.test(df$x_col, df$y_col, df$z_col)\nprint(partial_corr)`
    },
    "Crosstab / Contingency Table": {
        "Python": `crosstab = pd.crosstab(df['var1'], df['var2'])\nprint(crosstab)`,
        "R": `table(df$var1, df$var2)`
    },
    "Covariance Matrix": {
        "Python": `cov_matrix = df.cov()\nprint(cov_matrix)`,
        "R": `cov(df)`
    },
    "Association Rules": {
        "Python": `from mlxtend.frequent_patterns import apriori, association_rules\nfrequent_itemsets = apriori(df, min_support=0.07, use_colnames=True)\nrules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)`,
        "R": `library(arules)\nrules <- apriori(df, parameter = list(supp = 0.001, conf = 0.8))`
    },
    "Linear Regression": {
        "Python": `from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X, y)\nprint(model.coef_, model.intercept_)`,
        "R": `model <- lm(y ~ ., data=df)\nsummary(model)`
    },
    "Logistic Regression": {
        "Python": `from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()\nmodel.fit(X, y)\nprint(model.coef_, model.intercept_)`,
        "R": `model <- glm(y ~ ., data=df, family=binomial)\nsummary(model)`
    },
    "Polynomial Regression": {
        "Python": `from sklearn.preprocessing import PolynomialFeatures\npoly = PolynomialFeatures(degree=2)\nX_poly = poly.fit_transform(X)\nmodel = LinearRegression().fit(X_poly, y)`,
        "R": `model <- lm(y ~ poly(x, 2), data=df)\nsummary(model)`
    },
    "Ridge/Lasso Regression": {
        "Python": `from sklearn.linear_model import Ridge, Lasso\nridge = Ridge().fit(X, y)\nlasso = Lasso().fit(X, y)`,
        "R": `library(glmnet)\nridge <- glmnet(as.matrix(X), y, alpha=0)\nlasso <- glmnet(as.matrix(X), y, alpha=1)`
    },
    "Poisson Regression": {
        "Python": `import statsmodels.api as sm\n\n# Assuming 'X' are your features and 'y' is your count data target\npoisson_model = sm.Poisson(y, sm.add_constant(X)).fit()\nprint(poisson_model.summary())`,
        "R": `model <- glm(y ~ x1 + x2, data = df, family = poisson)\nsummary(model)`
    },
    "Multiple Regression": {
        "Python": `import statsmodels.api as sm

# Assuming 'X' are your multiple independent variables and 'y' is your dependent variable
X = sm.add_constant(X) # adding a constant

model = sm.OLS(y, X).fit()
print(model.summary())`,
        "R": `// Assuming 'my_data' is your dataframe
model <- lm(dependent_var ~ independent_var1 + independent_var2 + independent_var3, data = my_data)
summary(model)`
    },
    "Train/Test Split": {
        "Python": `from sklearn.model_selection import train_test_split\n# For features and labels\nX = df.drop('target', axis=1)\ny = df['target']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`,
        "R": `set.seed(123)\n# For features and labels\ntrain_idx <- sample(seq_len(nrow(df)), size = 0.8*nrow(df))\ntrain <- df[train_idx, ]\ntest <- df[-train_idx, ]`
    },
    "Cross Validation": {
        "Python": `from sklearn.model_selection import cross_val_score\nfrom sklearn.linear_model import LogisticRegression\n\n# Assuming 'X' are your features, 'y' is your target, and you're using a logistic regression model\nmodel = LogisticRegression()\nscores = cross_val_score(model, X, y, cv=5) /* 5-fold cross-validation */\nprint(f'Cross-validation scores: {scores}')\nprint(f'Average score: {scores.mean()}')`,
        "R": `library(caret)\n\n# Assuming 'df' is your dataframe and 'target' is your outcome variable\nctrl <- trainControl(method = \"cv\", number = 10) /* 10-fold CV */\nmodel <- train(target ~ ., data = df, method = \"glm\", trControl = ctrl)\nprint(model)`
    },
    "Classification (SVM, Decision Tree, Random Forest, KNN)": {
        "Python": `from sklearn.svm import SVC\nmodel = SVC()\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)`,
        "R": `library(e1071)\nmodel <- svm(y ~ ., data=train_data)\npreds <- predict(model, newdata=test_data)`
    },
    "Regression (Random Forest, SVR, etc.)": {
        "Python": `from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)`,
        "R": `library(randomForest)\nmodel <- randomForest(y ~ ., data=train_data)\npreds <- predict(model, newdata=test_data)`
    },
    "Clustering (KMeans, Hierarchical, DBSCAN)": {
        "Python": `from sklearn.cluster import KMeans\nmodel = KMeans(n_clusters=3)\nmodel.fit(df)`,
        "R": `kmeans_result <- kmeans(df, centers=3)\nprint(kmeans_result)`
    },
    "K-Means Clustering": {
        "Python": `from sklearn.cluster import KMeans\nimport matplotlib.pyplot as plt\n\n# Assuming 'X' are your features\nkmeans = KMeans(n_clusters=3, random_state=42)\ny_kmeans = kmeans.fit_predict(X)\n\nplt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')\ncenters = kmeans.cluster_centers_\nplt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);\nplt.title('K-Means Clustering')\nplt.show()`,
        "R": `// Assuming 'my_data' is your numeric data matrix/frame\nset.seed(123)\nkm_result <- kmeans(my_data, centers = 3, nstart = 25)\n\n# Print the results\nprint(km_result)\n\n# Visualize the clusters\nlibrary(factoextra)\nfviz_cluster(km_result, data = my_data)`
    },
    "Principal Component Analysis": {
        "Python": `from sklearn.decomposition import PCA\n\n# Assuming 'X' are your features\npca = PCA(n_components=2) // Reduce to 2 dimensions\nprincipalComponents = pca.fit_transform(X)\n\nprincipalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])\nprint(principalDf.head())`,
        "R": `// Assuming 'my_data' is your numeric data\n# The 'prcomp' function can be used\nresults <- prcomp(my_data, scale = TRUE)\n\n# Print results\nprint(results)\n\n# See the summary\nsummary(results)`
    },
    "Model Evaluation": {
        "Python": `from sklearn.metrics import accuracy_score, confusion_matrix\nacc = accuracy_score(y_test, preds)\ncm = confusion_matrix(y_test, preds)\nprint(acc, cm)`,
        "R": `library(caret)\nconfusionMatrix(preds, y_test)`
    },
    "Feature Importance": {
        "Python": `importances = model.feature_importances_\nprint(importances)`,
        "R": `importance <- varImp(model)\nprint(importance)`
    },
    "Time Series Decomposition": {
        "Python": `from statsmodels.tsa.seasonal import seasonal_decompose\nresult = seasonal_decompose(df['your_timeseries'], model='additive', period=12)\nresult.plot()`,
        "R": `library(forecast)\nts_data <- ts(df$your_timeseries)\ndecomp <- decompose(ts_data)\nplot(decomp)`
    },
    "Autocorrelation Plot": {
        "Python": `from pandas.plotting import autocorrelation_plot\nautocorrelation_plot(df['your_timeseries'])`,
        "R": `acf(df$your_timeseries)`
    },
    "ARIMA Modeling": {
        "Python": `from statsmodels.tsa.arima.model import ARIMA\nmodel = ARIMA(df['your_timeseries'], order=(1,1,1))\nmodel_fit = model.fit()\nprint(model_fit.summary())`,
        "R": `library(forecast)\nmodel <- auto.arima(df$your_timeseries)\nsummary(model)`
    },
    "Forecasting": {
        "Python": `from statsmodels.tsa.holtwinters import ExponentialSmoothing\nmodel = ExponentialSmoothing(df['your_timeseries']).fit()\nforecast = model.forecast(10)\nprint(forecast)`,
        "R": `library(forecast)\nmodel <- ets(df$your_timeseries)\nforecasted <- forecast(model, h=10)\nprint(forecasted)`
    },
    "PCA": {
        "Python": `from sklearn.decomposition import PCA\npca = PCA(n_components=2)\ncomponents = pca.fit_transform(df)\nprint(components)`,
        "R": `pca <- prcomp(df, scale. = TRUE)\nsummary(pca)`
    },
    "t-SNE": {
        "Python": `from sklearn.manifold import TSNE\ntsne = TSNE(n_components=2)\nX_embedded = tsne.fit_transform(df)`,
        "R": `library(Rtsne)\ntsne <- Rtsne(as.matrix(df))\nplot(tsne$Y)`
    },
    "UMAP": {
        "Python": `import umap\numap_model = umap.UMAP(n_components=2)\nembedding = umap_model.fit_transform(df)`,
        "R": `library(umap)\numap_result <- umap(df)\nplot(umap_result$layout)`
    },
    "Export DataFrame to CSV/Excel": {
        "Python": `# To CSV\ndf.to_csv('output.csv', index=False)\n\n# To Excel\ndf.to_excel('output.xlsx', index=False)`,
        "R": `# To CSV\nwrite.csv(df, 'output.csv', row.names=FALSE)\n\n# To Excel\nlibrary(writexl)\nwrite_xlsx(df, 'output.xlsx')`
    },
    "Generate Summary Report": {
        "Python": `summary = df.describe(include='all')\nprint(summary)`,
        "R": `summary(df)`
    }
};
