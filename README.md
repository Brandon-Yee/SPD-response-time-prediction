# SPD-response-time-prediction
A deep neural network approach to predicting Seattle police call response time.
- Data source: https://data.seattle.gov/Public-Safety/Call-Data/33kz-ixgy
- Seattle Police Beats shape file data: https://data-seattlecitygis.opendata.arcgis.com/datasets/seattle-police-beats-2018-present?geometry=-123.384%2C47.454%2C-121.298%2C47.778

## Thoughts on initial design:
### Features in the dataset:

- CAD Event Number: Unique ID

- Event Clearance Description: How the call was resolved, as reported by the primary officer.

- Call Type: How the call was received by the Communications Center.

- Priority: Priority of the call, as assigned by the CAD system.

- Initial Call Type: How the call was classified, initially by the Communication Center.

- Final Call Type: How the call was classified, finally by the primary officer.

- Original Time Queued: Time queued in the CAD system.

- Arrived Time: Time the first officer arrived on the call.

- Precinct: Precinct where the call originated.

- Sector: Sector where the call originated. All Sectors roll up to one of five Precincts.

- Beat: Beat where the call originated. All Beats roll up to Sectors.

### Network Design Task:
Thinking a feedforward network. Prediction target: 'Elasped Time before response' (time difference between 'Arrived Time' and 'Original Time Queued'.

- Train encoding for 'Initial Call Type' and again for 'Final Call Type'

### Data to use:
- 25 Jan 2018 to present (change in police beats went into effect on 24 Jan 2018)

## Data analysis tasks
- Visualization of precint locations and police precint/sector/beat
- Coloring of precint/sector/beat based on average calls per unit of time
- Coloring of precint/sector/beat based on average response times overall
- Plot of number of calls over time

## Data prep tasks
- Reduce down to data from 25 Jan 2018 to present
- Create train, validation, test sets
- Create pytorch data loaders

## Timeline
- Final Project Report due: Thursday, 10 June at 2pm
Report in form of conference paper
- Project Presentation: Tuesday, 8 June

