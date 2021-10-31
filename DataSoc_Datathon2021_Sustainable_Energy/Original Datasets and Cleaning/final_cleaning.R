### Using Real GDP - excluding years after 2016

# Libraries and Working Directory -----------------------------------------
library(dplyr) # General manipulation
library(readxl) # Reading WGI dataset
library(reshape2) # Reshaping WGI dataset
library(ggplot2) # Inspection of missing values
library(imputeTS) # Imputation package
setwd("C:/Users/Daniel/Desktop/DataSoc/UNSW x USYD Datathon/new")
options(scipen=999)

# Missing Data Visualiser ------------------------------------------------
perc_missing <- function(x){sum(is.na(x))/length(x)*100}

visualise <- function(df) {
  missing <- as.data.frame(apply(df, 2, perc_missing))
  colnames(missing) = "Percentage"
  missing["Variable"] <- rownames(missing)
  ggplot(missing, aes(x = Variable, y = Percentage, fill = Variable)) + 
    geom_bar(stat = "identity") + 
    labs(title = "Missing Data Distribution") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
          legend.position = "none")
}

# Energy Data - Cleaning ----------------------------------------------------
owid_data <- read.csv("owid-energy-data.csv")

### 1. Variable Selection (Change these fields as desired)
# Keeping other_renewable_exc_biofuel_electricity
temp <- owid_data$other_renewable_exc_biofuel_electricity

energy_1 <- owid_data %>%
  select(
    # Demographics (5)
    iso_code, country, year, gdp, population,
    # Electricity Generation (15)
    contains("electricity"),
    
    # Fossil Fuel Production (3)
    contains("production"),
    # Energy Consumption (13)
    contains("consumption"),
    # Electricity Consumption Share (12)
    contains("share_elec"),
    # Other metrics (4) Note: Energy per gdp calculated separately
    energy_cons_change_pct, energy_cons_change_twh,
    energy_per_gdp, energy_per_capita
  ) %>%
  select(
    # Removing variables contained in previous selection
    -contains("low_carbon"),
    -contains("fossil"),
    -contains("renewable")
  ) 

energy_1$other_renewable_exc_biofuel_electricity <- temp

energy_1 <- energy_1 %>%
  # Relocating columns 
  relocate(per_capita_electricity, .after = electricity_generation) %>%
  relocate(oil_electricity, .after = gas_electricity) %>%
  relocate(oil_consumption, .after = gas_consumption) %>%
  relocate(oil_share_elec, .after = gas_share_elec) %>%
  relocate(other_renewable_exc_biofuel_electricity, .after = wind_electricity)

### 2. Row Filtering 

# Apply function to count number of missing fields for each country (for Year 2000)
country_2000 <- energy_1 %>% 
  filter(year == 2000) %>%
  mutate(na_count = apply(., 1, function(x) sum(is.na(x))))

# Countries selected dependent on number of NAs <= 3
country_list <- as.vector(country_2000 %>% filter(na_count <= 3, country != "World",
                                                  country != "Taiwan", country != "Venezuela") %>% select(country))

## Filtering rows based on country list and year
energy_2 <- energy_1 %>%
  # Keeping only countries contained in country_list
  filter(country %in% country_list$country) %>%
  # Years between 2000 and 2019
  filter(year >= 2000 & year <= 2016)


# WGI Data - Cleaning -----------------------------------------------------

## Create empty list to populate with World Governance Indicators
WGI_list <- list()
i = 1
for (WGI in c("VoiceandAccountability", "Political StabilityNoViolence", "GovernmentEffectiveness", 
              "RegulatoryQuality", "RuleofLaw", "ControlofCorruption")) {
  temp <- read_excel("wgidataset.xlsx", WGI, skip = 14)
  # Extract only estimate values
  estimates <- c(1, 2, seq(from = 3, to = 134, by = 6))
  temp <- temp[, estimates]
  colnames(temp) <- c("country", "iso_code", c(seq(from = 1996, to = 2002, by = 2), seq(from = 2003, to = 2020)))
  temp <- temp %>%
    melt(., id = c("country", "iso_code"))
  colnames(temp)[3] = "year"
  colnames(temp)[4] = WGI
  WGI_list[[i]] = temp
  i = i + 1
}

## Extract all estimates of each WGI into a single dataframe
WGI_data <- WGI_list[[1]]
for (i in 2:6) {
  WGI_data <- merge(WGI_data, WGI_list[[i]], by = c("country", "iso_code", "year"))
}

# Converting year to integer, values to numeric
WGI_data$year = as.integer(as.numeric(as.character(WGI_data$year)))
WGI_data[,c(4:9)] <- sapply(WGI_data[,c(4:9)], as.numeric)

# Rename countries to match energy data
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Czech Republic", "Czechia")
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Slovak Republic", "Slovakia") 
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Egypt, Arab Rep.", "Egypt")
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Hong Kong SAR, China", "Hong Kong")
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Iran, Islamic Rep.", "Iran")
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Russian Federation", "Russia")
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Korea, Rep.", "South Korea")
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Taiwan, China", "Taiwan")
WGI_data$country <- replace(WGI_data$country, WGI_data$country == "Venezuela, RB", "Venezuela")
WGI_data$iso_code <- replace(WGI_data$iso_code, WGI_data$iso_code == "ROM", "ROU")

# Imputing Merged Datasets ------------------------------------------------
merged <- merge(WGI_data, energy_2, by = c("country", "iso_code", "year"), all.y = TRUE)

# Change production (if missing) to 0
na_to_zero <- function(x) {
  x[is.na(x)] <- 0
  return(x)
}

merged[,"coal_production"] <- na_to_zero(merged$coal_production)
merged[,"gas_production"] <- na_to_zero(merged$gas_production)
merged[,"oil_production"] <- na_to_zero(merged$oil_production)

# Imputing WGI missing values (in year 2001) with Moving Average (k = 1)
for (i in 4:9) {
  merged[,i] <- na_ma(merged[,i], k = 1)
}

# Exporting Datasets ------------------------------------------------------

WGI_columns <- c("VoiceandAccountability", "Political StabilityNoViolence", "GovernmentEffectiveness", 
                 "RegulatoryQuality", "RuleofLaw", "ControlofCorruption")

# Separating merged dataset into two
WGI <- merged %>%
  select(country, iso_code, year, all_of(WGI_columns))

Energy <- merged %>%
  select(-all_of(WGI_columns))

write.csv(merged, "energy_WGI-new.csv", row.names = FALSE)
write.csv(WGI, "WGI-new.csv", row.names = FALSE)
write.csv(Energy, "energy-new.csv", row.names = FALSE)

# Given the two separate datasets, they should merge to create "energy_WGI.csv"
test <- merge(WGI, Energy, by = c("country", "iso_code", "year")) %>%
  relocate(gdp, .after = year)
visualise(WGI)
