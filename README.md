# B211---Assignment-4
The purpose of this program is to start performing numeric data analysis using Python and the Scipy module, using real world data. 

The project was designed around the dataset given via kaggle.com. This allowed the dataframe to remain relatively consistent through the blocks of analytical code. The class design of dataframe can be detailed as such:
1. Data Modifier: loads and cleans dataset
2. Filtering: isolates specific stages
3. Individual Analysis: Focuses on a single players trajectory using regression and interpolation
4. Aggregate Analysis: Preforms population level statistics and T-tests.

Class attributes included the following:
os.path = identifies the directory to allow for smooth program usage
df = the primary dataframe containing the dataset
regression model = a dictionary storing the regression values such as slope and R^2

Class methods include the following:
run_population_tests(): Executes T-tests and caluclates higher order moments.
interopolate_missing_years(): use interpid to backfill some stats
analyze_player_accuracy(): calculates 3PM/3PA and fits a linear regression model

Limitations were noted during the process of coding the program such as linearity in which preformance is modeled strictly in a linear fashion, extrapolation is typically unreliable as these estimates may not hold weight in the actual world. 
