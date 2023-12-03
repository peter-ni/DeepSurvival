library(tidyverse)


# Read in Data
data_clinical_sample = read_csv("~/Desktop/DataSelector/data_clinical_sample_out.csv")
data_clinical_patient = read_csv("~/Desktop/DataSelector/data_clinical_patient_out.csv")
data_cna = read_csv("~/Desktop/DataSelector/data_cna_out.csv", skip = 1) %>% mutate(PATIENT_ID_TRIM = substr(PATIENT_ID,0,9))



# Non-Small Cell Lung Cancer Dataset (5K)

data_clinical_sample_nsclc = data_clinical_sample %>% filter(CANCER_TYPE == "Non-Small Cell Lung Cancer")
nsclc_patient_ids = data_clinical_sample_nsclc$PATIENT_ID


data_clinical_patient_nsclc = data_clinical_patient %>% filter(PATIENT_ID %in% nsclc_patient_ids)
data_cna_nsclc = data_cna %>% filter(PATIENT_ID_TRIM %in% nsclc_patient_ids)

select_clin_sample = data_clinical_sample_nsclc %>% transmute(PATIENT_ID = PATIENT_ID,
                                                              SAMPLE_TYPE = SAMPLE_TYPE,
                                                              SAMPLE_COVERAGE = SAMPLE_COVERAGE,
                                                              TUMOR_PURITY = TUMOR_PURITY,
                                                              MSI_SCORE = MSI_SCORE,
                                                              TMB_NONSYNONYMOUS = TMB_NONSYNONYMOUS)
select_clin_patient = data_clinical_patient_nsclc %>% transmute(PATIENT_ID = PATIENT_ID,
                                                                SEX = SEX,
                                                                RACE = RACE,
                                                                OS_STATUS = OS_STATUS,
                                                                AGE_AT_EVIDENCE_OF_METS = AGE_AT_EVIDENCE_OF_METS,
                                                                AGE_AT_SEQUENCING = AGE_AT_SEQUENCING,
                                                                AGE_AT_DEATH = AGE_AT_DEATH,
                                                                AGE_AT_LAST_CONTACT = AGE_AT_LAST_CONTACT)

select_cna = data_cna_nsclc %>% select(-PATIENT_ID) %>% rename(PATIENT_ID = PATIENT_ID_TRIM)


noresponse_list = c("Pt refused to answer", "Unknown", "No value entered")
nonwhite_list = data_clinical_patient_nsclc$RACE %>% as.factor() %>% levels() %>% setdiff(noresponse_list) %>% setdiff("White")

unified_dataset = inner_join(select_clin_patient, select_clin_sample, by = join_by(PATIENT_ID)) %>% 
  inner_join(select_cna, by = join_by(PATIENT_ID)) %>% mutate(STATUS = ifelse(OS_STATUS == "1:DECEASED", 1, 0),
                                                              TIME = ifelse(STATUS == 1, AGE_AT_DEATH - AGE_AT_SEQUENCING, AGE_AT_LAST_CONTACT - AGE_AT_SEQUENCING),
                                                              SEX = ifelse(SEX == "MALE", 0, 1),
                                                              RACE = ifelse(RACE == "White", 1, RACE),
                                                              RACE = ifelse(RACE %in% nonwhite_list, 0, RACE),
                                                              RACE = ifelse(RACE %in% noresponse_list, -1, RACE),
                                                              SAMPLE_TYPE = ifelse(SAMPLE_TYPE == "Metastasis", 1, 0),
                                                              TIME_SINCE_DX = AGE_AT_SEQUENCING - AGE_AT_EVIDENCE_OF_METS,
                                                              TIME_SINCE_DX = ifelse(TIME_SINCE_DX <= 0 | is.na(TIME_SINCE_DX), -1, TIME_SINCE_DX),
                                                              TUMOR_PURITY = ifelse(is.na(TUMOR_PURITY), -1, TUMOR_PURITY)) %>% 
  select(-OS_STATUS) %>% select(-AGE_AT_DEATH) %>% select(-AGE_AT_LAST_CONTACT) %>% select(-AGE_AT_EVIDENCE_OF_METS)

unified_dataset = unified_dataset %>% filter(!is.na(TIME)) %>% filter(TIME > 0) %>% mutate(PATIENT_ID = row_number())

# Name Assignments

cna_names = data_cna_nsclc %>% colnames %>% setdiff(c("PATIENT_ID","PATIENT_ID_TRIM"))
clin_names = unified_dataset %>% colnames %>% setdiff(cna_names) %>% setdiff(c("PATIENT_ID","TIME","STATUS"))



# Output cleaned CSV data and names
cancer_data = unified_dataset
file_path = '/Users/peterni/Desktop/Projects/DeepSurvival/DeepSurvival/cleaned_datasets/cancer_data.csv'
write.csv(cancer_data, file = file_path, row.names = FALSE)









