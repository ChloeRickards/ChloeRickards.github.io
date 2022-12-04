/* The following script creates a table displaying the cumulative global COVID-19 cases, deaths, and Case Fatality Rate (CFR)
It uses the "Daily cases and deaths by date reported to WHO" table found in https://covid19.who.int/data
for cases and deaths up to the current date. Then, it finds the Case Fatality Rate (CFR) by dividing deaths by cases. 
The CFR can be thought of as the likelihood of a COVID-19 case eventually dying from COVID-19, or, the fraction of cases
that end up dying.

Author's note: This is considered a "crude" calculation of the Case Fatality Rate because it
does not take the timeline of cases and deaths into account and because it is derived from
cumulative cases and deaths.

*/


-- Select relevant columns from the WHO data. Only the most recent records are kept.
SELECT Country, Cumulative_cases AS Cases, Cumulative_deaths AS Deaths
INTO CFRs FROM [WHO-COVID-19-global-data] AS t1
WHERE t1.Date_reported = (SELECT MAX(t2.Date_reported)
                 FROM [WHO-COVID-19-global-data] AS t2
                 WHERE t2.Country = t1.Country);
GO

-- add a column to the table
ALTER TABLE CFRs
ADD Crude_Cumul_CFR float;
GO

-- Calculate the CFR and update the table. Cases and deaths are reported as integers,
-- and they need to be typecast to floats in order for division to proceed
UPDATE CFRs  
SET Crude_Cumul_CFR = ROUND((CAST(Deaths AS float)/CAST(Cases AS float)), 4)
WHERE Deaths > 0
AND Cases > 0;
GO

-- Display the table, and show the CFRs as percentages.
SELECT Country, Cases, Deaths, CONVERT(VARCHAR(10), Crude_Cumul_CFR*100) + '%' AS Crude_Cumul_CFR_percent
FROM CFRs
ORDER BY Crude_Cumul_CFR DESC