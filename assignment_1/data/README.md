### All datasets are obtained from [Inside Airbnb](https://insideairbnb.com), which provides publicly available scraped data on Airbnb listings.

## Core dataset
- **City:** Barcelona, Spain  
- **Snapshot:** Q1 2025  
- **Source:**  
  https://data.insideairbnb.com/spain/catalonia/barcelona/2025-03-05/data/listings.csv.gz  

This dataset is used for model training and initial evaluation.

## Validity datasets
Two additional “live” datasets are used to assess model validity across time and location:

1. **Later date (time validity)**  
   - City: Barcelona  
   - Snapshot: Q3 2025  
   - Source:  
     https://data.insideairbnb.com/spain/catalonia/barcelona/2025-09-14/data/listings.csv.gz  

2. **Other city (geographic validity)**  
   - City: Sevilla, Spain  
   - Snapshot: Q3 2025  
   - Source:  
     https://data.insideairbnb.com/spain/andaluc%C3%ADa/sevilla/2025-09-29/data/listings.csv.gz  

## Notes on reproducibility
The raw data files are not stored directly in this repository due to their size.  
All datasets are downloaded programmatically within the analysis notebooks using the URLs above, ensuring full reproducibility of the workflow.
