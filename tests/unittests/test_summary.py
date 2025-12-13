import numpy as np
import pandas as pd

from jaxstar.utils.summary import (
    summary_hdi,
    summary_pct,
    summary_mean,
    summary_stats,
    summarize_results,
)


def test_summary_percentile_and_mean():
    data = np.array([0.0, 1.0, 2.0, 3.0])

    val, upp, low = summary_pct(data)
    assert val == np.percentile(data, 50)
    assert np.isclose(upp, np.percentile(data, 84) - val)
    assert np.isclose(low, val - np.percentile(data, 16))

    mu, sigma_upp, sigma_low = summary_mean(data)
    assert np.isclose(mu, data.mean())
    assert np.isclose(sigma_upp, data.std())
    assert np.isclose(sigma_low, data.std())


def test_summary_hdi_matches_interval_width():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=1.0, scale=0.5, size=1000)
    val, upp, low = summary_hdi(samples, prob=0.68, peak_prob=0.2)
    # Peak value should be near the mean; error bars should be positive
    assert 0.5 < val < 1.5
    assert upp > 0
    assert low > 0


def test_summary_stats_and_summarize_results(tmp_path):
    postdir = tmp_path / "post"
    postdir.mkdir()
    filename = postdir / "123_samples.csv"
    df = pd.DataFrame({"kmag": [9.5, 10.5, 11.5], "teff": [5800, 5850, 5900]})
    df.to_csv(filename, index=False)

    summary = summary_stats(
        postdir=str(postdir) + "/",
        names=["123"],
        keys=["kmag", "teff"],
        stat="pct",
    )
    assert list(summary.columns) == [
        "name",
        "iso_kmag",
        "iso_kmag_upp",
        "iso_kmag_low",
        "iso_teff",
        "iso_teff_upp",
        "iso_teff_low",
    ]
    assert summary.loc[0, "name"] == "123"

    dinput = pd.DataFrame(
        {
            "kepid": ["123"],
            "kmag": [10.0],
            "kmag_err": [0.2],
            "teff": [5850],
            "teff_err": [50],
            "binflag": [0],
        }
    )
    merged = summarize_results(
        postdir=str(postdir) + "/",
        dinput=dinput,
        keys=["kmag", "teff"],
        obskeys=["kmag", "teff"],
        stat="pct",
    )
    # Ensure residuals are computed and columns exist with expected values
    assert merged.shape[0] == 1
    assert np.isclose(merged.loc[0, "dkmag"], summary.loc[0, "iso_kmag"] - 10.0)
    assert np.isclose(merged.loc[0, "dsigmakmag"], merged.loc[0, "dkmag"] / 0.2)
    assert np.isclose(merged.loc[0, "dteff"], summary.loc[0, "iso_teff"] - 5850)
    assert np.isclose(merged.loc[0, "dsigmateff"], 0.0)
    assert np.isclose(merged.loc[0, "dsigmaobs"], merged.loc[0, "dsigmakmag"])
