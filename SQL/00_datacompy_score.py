from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import DataCompyScore
import asyncio


async def main():
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
10000001234,123.45,George Maharis,14530.1555,2017-01-01
10000001235,0.45,Michael Bluth,1,2017-01-01
10000001236,1345,George Bluth,,2017-01-01
10000001237,123456,Bob Loblaw,345.12,2017-01-01
10000001238,1.05,Lucille Bluth,,2017-01-01
10000001238,1.05,Loose Seal Bluth,,2017-01-01
"""

    data2 = """acct_id,dollar_amt,name,float_fld
10000001234,123.4,George Michael Bluth,14530.155
10000001235,0.45,Michael Bluth,
10000001236,1345,George Bluth,1
10000001237,123456,Robert Loblaw,345.12
10000001238,1.05,Loose Seal Bluth,111
"""

    sample = SingleTurnSample(response=data1, reference=data2)
    # row-wise F1 (default): no rows match across the two CSVs in the docs example,
    # so we demo with mode="columns" + metric="recall" which yields a non-NaN score
    # while still exercising the metric. Switch to DataCompyScore() for the default
    # rows / f1 behaviour.
    scorer = DataCompyScore(mode="columns", metric="recall")
    score = await scorer.single_turn_ascore(sample)
    print(f"Recall (column-wise): {score}")


if __name__ == "__main__":
    asyncio.run(main())