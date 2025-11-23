
# %% [code]

import os
# DO NOT COMMIT THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

import dataclasses
import pandas as pd
from openai import OpenAI
from utils import parse_survey_data

@dataclasses.dataclass
class Scenario:
    name: str
    initial_belief: str
    final_belief: str
    ground_truth: str
    before_column: str
    after_column: str

scenarios = [
    Scenario(
        name='CLOSE',
        initial_belief="The AV stopped because of the pickup/drop-off zone ahead. Therefore, if the AV was slightly to the left, it would not change its behavior; it would still stop.",
        final_belief="The AV stopped because of the parked cars nearby to the right. If it was further to the left, it would be further from the parked cars, and therefore it would keep moving.",
        ground_truth="The AV stopped because of the parked cars nearby to the right. If it was further to the left, it would be further from the parked cars, and therefore it would keep moving.",
        before_column="CLOSE before",
        after_column="CLOSE after",
    ),
    Scenario(
        name='ASV',
        initial_belief="The AV stopped because of the traffic cone. Therefore, if the traffic cone was not there, it would have kept moving.",
        final_belief="The AV stopped because it hallucinated a stopped vehicle, not because of the traffic cone. Therefore, if the traffic cone was not there, it may have still stopped." ,
        ground_truth="The AV stopped because it hallucinated a stopped vehicle, not because of the traffic cone. Therefore, if the traffic cone was not there, it would have still stopped.",
        before_column="ASV before",
        after_column="ASV after",
    ),
    Scenario(
        name='BIKE',
        initial_belief="The AV can detect and stop for cyclists reliably, which is why it stopped for the cyclist. Therefore, we can trust that the AV will keep stopping reliably for cyclists in the future.",
        final_belief="The AV cannot detect and stop for cyclists reliably. Therefore, we cannot trust that the AV will keep stopping reliably for cyclists in the future.",
        ground_truth="The AV cannot detect and respond to cyclists at all; instead, it stopped because of the automatic emergency braking system. Therefore, we cannot trust that the AV will keep stopping reliably for cyclists in the future.",
        before_column="BIKE before",
        after_column="BIKE after",
    )
]


#df = parse_survey_data("data/motional_drivers.csv", skip_rows=3)
df = parse_survey_data("data/normal_people_new.csv", skip_rows=4)

client = OpenAI()
        
# %% [code]

results = []

for index, row in df.iterrows():

    passenger_data = {
        'name': row['participant name'],
    }

    for scenario in scenarios:
        
        responses = []
        for anchor in [scenario.initial_belief, scenario.final_belief, scenario.ground_truth]:

            response = client.responses.create(
                model="gpt-5",
                #reasoning={"effort": "low"},
                instructions="""
                    Imagine you're a human subject in a psychological study.
                    You are given a anchor mental model and prediction (denoted by ANCHOR) for the behavior of an autonomous vehicle (AV)
                    and two candidate explanations and predictions for AV behavior (denoted by CANDIDATE 1 and CANDIDATE 2) given by passengers in the AV.
                    Please indicate which of the two candidate explanations and predictions are most consistent with the anchor mental model and prediction.
                    Respond simply with the number of the candidate (1 or 2).
                    """,
                input=f"""
                ANCHOR: {anchor}
                CANDIDATE 1: {row[scenario.before_column]}
                CANDIDATE 2: {row[scenario.after_column]}
                """
            )
            
            print(f"Scenario: {scenario.name}, Anchor: {anchor}, Passenger: {row['participant name']}")
            print(f"Response: {response.output_text}")
            responses.append(response.output_text)
        
        passenger_data[scenario.name] = ' '.join(responses)
        
    results.append(passenger_data)
            

results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)

# %% [code]