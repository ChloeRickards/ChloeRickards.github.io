/* This script creates a view of COVID-19 vaccinations along with social, environmental, and economic characteristics
for each ZIP code in California. 

Vaccination information is compiled by ZIP code, while the envirosocioeconomic factors are compiled via census tract.
Tracts are matched to ZIP codes in order to join the tables together. Final values are averaged across ZIP codes

Author's Note: The CES is a score that takes environmental and socioeconomic factors together to determine
how vulnerable a population is to pollution. I chose this dataset because it includes socioeconomic factors
like poverty and unemployment which are known to negatively impact outcomes of COVID-19 cases. And, I was
curious to see if/how environmental factors relate to COVID outcomes and vaccinations.

Sources:
Tract to ZIP: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
COVID vaccinations: https://covid19.ca.gov/vaccination-progress-data/#zip-code
CalEnviroScreen: https://data.ca.gov/dataset/calenviroscreen-2-0

*/

-- Join the tables together
CREATE VIEW CovidEquity
AS
SELECT CAinfo.zip, covid.county, covid.percent_of_population_fully_vaccinated,
		covid.percent_of_population_with_1_plus_dose, calenviro.CES_2_0_Score,
		calenviro.Pollution_Burden_Pctl, calenviro.Education_Pctl, calenviro.Poverty_Pctl, calenviro.Unemployment_Pctl
FROM TRACT_ZIP_122021 AS CAinfo
	LEFT JOIN
	covid19vaccinesbyzipcode_test AS covid
	ON CAinfo.zip = covid.zip_code_tabulation_area
	LEFT JOIN
	calenviroscreen-final-report AS calenviro
	ON CAinfo.tract = calenviro.Census_Tract
WHERE CAinfo.usps_zip_pref_state = 'CA';
GO

-- Because there are more census tracts than ZIP codes, we condense entries by taking the average
-- across the ZIP code and the county
SELECT
	zip AS ZIP,
	county AS County,
	AVG(percent_of_population_fully_vaccinated)*100 AS Percent_Fully_Vaccinated,
	AVG(percent_of_population_with_1_plus_dose)*100 AS Percent_At_Least_1_Dose,
	AVG(CES_2_0_Score) AS CES_2, -- higher CES = more vulnerable population
	AVG(Pollution_Burden_Pctl) AS Pollution_Burden_Percentile,
	AVG(Education_Pctl) AS Education_Percentile,
	AVG(Poverty_Pctl) AS Poverty_Percentile,
	AVG(Unemployment_Pctl) AS Unemployment_Percentile
FROM CovidEquity
WHERE county IS NOT NULL AND percent_of_population_fully_vaccinated IS NOT NULL
GROUP BY Zip, County
ORDER BY CES_2 DESC;