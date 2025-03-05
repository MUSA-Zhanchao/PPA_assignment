# Midterm assignment instruction

Congratulations! You've been hired as an intern at Marie-Antoinette Predictions (MAP) Inc., a company specializing in **spatial predictive modeling**. Your first task? **Build a predictive model for home prices in Charlotte, NC**!

You’ve been given a dataset, `studentData.geojson`, containing **property sales data** from Charlotte. However, home prices aren’t just about square footage—**location matters**!

Your job is to **improve the prediction model** by integrating **locational characteristics** from external data sources (e.g., open data portals for Charlotte). The intern with the **lowest RMSE** will be celebrated! 🎉

## Assignment Requirements
### 1. Data Collection & Exploration
1. **Load & Clean Data**:
- Import studentData.geojson.
- Check for missing values and outliers.

2. **Feature Engineering**:
- Transform skewed variables if necessary (e.g., `log(price)`).
- Create **new spatial features** (distance to parks, crime rates, etc.).
- Consider adding **interaction terms** (e.g., house size × land area), **dummy variables, and engineered categorical variables** (castle!).

3. **Summary Statistics & Correlations**:
- Generate a **correlation matrix** to explore relationships.
- Include **four scatterplots** of home prices vs. key predictors (including at least **two new spatial variables**).

4. **Mapping**:
- Create **one map of home prices**.
- Create **three maps of independent variables** (e.g., number of bedrooms, crime rates, school quality).

## 2. Model Building & Evaluation

1. **Train-Test Split**
- Split data into **training (80%) and testing (20%) sets**.

2. **Modeling: Iteratively work on running your OLS model - adding variables and evaluating diagnostics. We don't have to see every iteration! But for your final model:**
- Discuss coefficients and model fit.
- Show and interpret basic model diagnostics
- Report **Mean Absolute Error (MAE), RMSE, and R²**.

## 3. Spatial Analysis & Residual Diagnostics (🚨 Covering in Class 3/7)

1. **Residual Analysis:**

- Plot **predicted vs. observed prices**.
- Map **residuals by neighborhood** to check for spatial bias.

2. **Spatial Autocorrelation of Residuals**

- Compute **Moran’s I** to test for **spatial autocorrelation**.
- If Moran’s I is significant, discuss **why location affects residuals**.
- Explore potential improvements to address spatial dependence.

3. **Interpretation & Limitations**

- Discuss where your model performs best and worst.

## Presentation & Markdown Quality
Your report should be a **well-organized and visually appealing** R Markdown document:

- Use **clear section headers**.
- Ensure all **code is annotated with explanations**.
- Provide **interpretations in your own words** for all results.
- Use **well-labeled plots and tables**.
- Make your markdown **easy to read and structured well**.

# Rubric

![Screenshot 2025-03-02 at 10 05 59 PM](https://github.com/user-attachments/assets/3a4fe934-e0cb-4537-8615-550d880b0d88)

# FAQ:

### Q: May I work together with a friend?

A: Yes, absolutely! However, you should each turn in your own markdown with your own writeup. Also, indicate who you worked with.

### Q: This isn't due until the Friday after spring break, so can I wait until the day before to start and then just ask for an extension?

A: No. This due date is THE Due Date. Plan accordingly.

### Q:  The data are for all of Mecklenburg County, can I limit it to just Charlotte?

A: Yes, please just indicate what decisions you made and justify them. If you do too much fiddling and restrict your dataset so much to win the prediction competition, your model will likely be Overfit, and you'll fail in the end!

### Q: I'm lost and have no idea what to do or what you mean by your instructions. Or there is a big error somewhere that we all see...what to do?

A: Come to my office hours. Email me if there is a problem.

### Q: This doesn't feel like a midterm because it's so much fun. Is that OK?

A: Definitely.

### Q: Did your friend ChatGPT help with creating this assignment?

A: A little - but I have thoroughly revised it, and all remaining errors are my own :)

# Rubric Continue

![Screenshot 2025-03-02 at 10 09 08 PM](https://github.com/user-attachments/assets/a6a805d7-3bd4-4e48-9f8b-fe3da8b368f4)
![Screenshot 2025-03-02 at 10 09 36 PM](https://github.com/user-attachments/assets/f1355963-1cd2-4ad3-bea7-38370144e697)
