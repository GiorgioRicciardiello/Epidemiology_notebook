from typing import Union, Optional, Any
import numpy as np
from scipy.stats import norm


class SampleSizeEstimation:
    def __init__(self, feature_type: str,
                 output_type: str,
                 alpha: float,
                 beta: float,
                 comparison: str = 'two sided'):
        self.feature_type: str = feature_type
        self.output_type: str = output_type
        self.alpha = alpha
        if alpha < 0 or alpha > 1:
            raise ValueError(f'alpha values must be between [0, 1]. Given {alpha}')
        self.beta = beta
        if beta < 0 or beta > 1:
            raise ValueError(f'alpha values must be between [0, 1]. Given {alpha}')
        # if comparison == 'two sided':
        #     self.alpha: float = self.alpha / 2
        #     self.beta: float = self.beta / 2
        #     self.two_sided: bool = True

        self.suggested_calculation()

    def suggested_calculation(self):
        if self.feature_type == 'dichotomous' and self.output_type == 'continuous':
            print(f'Call the method - self.in_dichotomous_out_continuous()')
        if self.feature_type == 'dichotomous' and self.output_type == 'dichotomous':
            print(f'Call the method - self.in_dichotomous_out_dichotomous()')

    def in_dichotomous_out_dichotomous(self,
                                       r: Union[int, float],
                                       effect: Optional[Union[int, float]] = None,
                                       p1: Union[int, float] = None,
                                       p0: Union[int, float] = None,
                                       study_type: str = 'other',
                                       odds_ratio: Optional[float] = None,
                                       risk_ratio: Optional[float] = None) -> int:
        """

        :param p1: int | float, proportion of subjects expected to have the outcome in one group
        :param p0: int | float, proportion expected in the other group
        :param r: ratio of unexposed to exposed or controls to cases
        :param effect:int | float, effect size
        :param study_type: srt, ['case-control' or other]
        :param odds_ratio: float, if we have odds_ratio we can get p1 OR p0
        :param risk_ratio; float, risk ratio to compute p1 or po
        :return:
        """
        z_alpha = self.calculate_z_values(alpha=self.alpha,
                                          comparison='two sided')
        z_power = self.calculate_z_values(alpha=self.beta,
                                          comparison='right sided')
        if study_type == 'case-control':
            # if odds_ratio is not None and risk_ratio is not None:
            #     raise ValueError(f'Only risk or odds can be considered')
            # if odds_ratio is not None:
            #     if p1 is None:
            #         p1 = self.get_proportions_from_odds(odds_ratio=odds_ratio, p0=p0)
            #     if p0 is None:
            #         p0 = self.get_proportions_from_odds(odds_ratio=odds_ratio, p1=p1)
            # if risk_ratio is not None:
            #     if p1 is None:
            #         p1 = self.get_proportions_from_risk(risk_ratio=risk_ratio, p0=p0)
            #     if p0 is None:
            #         p0 = self.get_proportions_from_risk(risk_ratio=risk_ratio, p1=p1)
            effect_, p1, p0 = self.get_proportions(odds_ratio=odds_ratio,
                                                   risk_ratio=risk_ratio,
                                                   p0=p0,
                                                   p1=p1)
            if effect is None:
                effect = effect_

            p_hat = (p1 + r * p0) / (1 + r)
            num_n = (z_alpha + z_power) ** 2 * p_hat * (1 - p_hat) * (r + 1)
            den_n = effect ** 2 * r
        else:
            p_hat = (p1 + r * p0) / (1 + r)
            num_n = (z_alpha + z_power) ** 2 * p_hat * (1 - p_hat) * (r + 1)
            den_n = (effect ** 2) * r
        sample_size = np.ceil(num_n / den_n).astype(int)
        print(f'\t z_alpha: {z_alpha}\n\t z_power: {z_power}\n\t p_hat: {p_hat}\n\t p0 {p0}\n\t p1 {p1}\n')
        print(f'Sample size { sample_size}')
        return sample_size

    def in_dichotomous_out_continuous(self, effect: Union[int, float],
                                      std_dev: Union[int, float],
                                      r: Union[int, float] = 1) -> int:
        """

        :param effect:int | float, effect size
        :param std_dev: standard deviation
        :param r: ratio of unexposed to exposed or controls to cases
        :return:
        """
        # if self.two_sided:
        z_alpha = self.calculate_z_values(alpha=self.alpha, comparison='two sided')
        z_power = self.calculate_z_values(alpha=self.beta, comparison='right sided')
        # print(f'z_alpha: {z_alpha}; \t z_power: {z_power}')
        num_n = (r + 1) * (z_alpha + z_power) ** 2 * std_dev ** 2
        den_n = effect ** 2 * r
        sample_size = np.ceil(num_n / den_n).astype(int)
        print(f'Sample size {sample_size}')
        return sample_size

    def get_power_from_sample_size(self, difference: str,
                                   sample_size: int,
                                   r: Union[int, float] = 1,
                                   p1: Optional[Union[int, float]] = None,
                                   p0: Optional[Union[int, float]] = None,
                                   effect: Union[int, float] = None,
                                   sigma: Optional[Union[int, float]] = None,
                                   odds_ratio: Optional[Union[int, float]] = None,
                                   risk_ratio: Optional[float] = None
                                   ) -> Union[int, float]:
        """
           Solve for power using the sample size.
        In the proportions, p0 or p1 can be extracted from the effect (if it's the odds ratio)
        :param difference: str, difference ['means', 'proportions']
        :param p1: int | float, proportion of subjects expected to have the outcome in one group
        :param p0: int | float, proportion expected in the other group
        :param effect:int | float, effect size
        :param sigma:
        :param sample_size:
        :param r: ratio of unexposed to exposed or controls to cases
        :param odds_ratio: float, if we have odds_ratio we can get p1 OR p0
        :param risk_ratio; float, risk ratio to compute p1 or po
        :return:
        """
        z_alpha = self.calculate_z_values(alpha=self.alpha, comparison='two sided')
        n = sample_size
        if difference not in ['means', 'proportions']:
            raise ValueError(f'Difference must be means or proportions, not {difference}')

        if difference == 'means':
            if sigma is None:
                raise ValueError(f'Specify Sigma for difference in means')
            return (effect / sigma) * np.sqrt((n * r) / (r + 1)) - z_alpha

        elif difference == 'proportions':
            # if odds_ratio is not None and risk_ratio is not None:
            #     raise ValueError(f'Only risk or odds can be considered')
            # if odds_ratio is not None:
            #     if p1 is None:
            #         p1 = self.get_proportions_from_odds(odds_ratio=odds_ratio, p0=p0)
            #     if p0 is None:
            #         p0 = self.get_proportions_from_odds(odds_ratio=odds_ratio, p1=p1)
            # if risk_ratio is not None:
            #     if p1 is None:
            #         p1 = self.get_proportions_from_risk(risk_ratio=risk_ratio, p0=p0)
            #     if p0 is None:
            #         p0 = self.get_proportions_from_risk(risk_ratio=risk_ratio, p1=p1)
            effect_, p1, p0 = self.get_proportions(odds_ratio=odds_ratio,
                                                   risk_ratio=risk_ratio,
                                                   p0=p0,
                                                   p1=p1)
            if effect is None:
                effect = effect_
            p_hat = (p1 + r * p0) / (1 + r)
            num = n * effect ** 2 * r
            den = (r + 1) * p_hat * (1 - p_hat)
            z_power = np.sqrt(num / den) - z_alpha
            print(f'Z_power: {z_power}')
            beta = 1 - norm.cdf(z_power)
            print(f'Power: {100-beta*100}')
            return z_power

    def get_proportions(self, risk_ratio: Optional[float] = None,
                        odds_ratio: Optional[float] = None,
                        p1: Optional[float] = None,
                        p0: Optional[float] = None) -> tuple[Union[float, Any], Union[float, Any], Optional[Any]]:
        """
         Get the proportions from the Odds Ratio or Risk Ratio
        :param risk_ratio: float, if the risk ratio is given
        :param odds_ratio: float, if the odds ratio is given
        :param p1: float,
        :param p0: float,
        :return:
            effect size, p1, p0
        """
        if odds_ratio is not None and risk_ratio is not None:
            raise ValueError(f'Only risk or odds can be considered')
        if p1 is None and p0 is None:
            raise ValueError(f'At least one p1 or p0 should be not None')
        if odds_ratio is not None:
            if p1 is None:
                p1 = self.get_proportions_from_odds(odds_ratio=odds_ratio, p0=p0)
            if p0 is None:
                p0 = self.get_proportions_from_odds(odds_ratio=odds_ratio, p1=p1)
        if risk_ratio is not None:
            if p1 is None:
                p1 = self.get_proportions_from_risk(risk_ratio=risk_ratio, p0=p0)
            if p0 is None:
                p0 = self.get_proportions_from_risk(risk_ratio=risk_ratio, p1=p1)
        effect = p1 - p0
        return effect, p1, p0

    def get_proportions_from_odds(self, odds_ratio: float, p0: Optional[float] = None,
                                  p1: Optional[float] = None) -> float:
        """
        Get a proportion from the odds ratio
        If p1 is given, it returns po
        If p0 is given, it returns p1
        :param odds_ratio: float, odds ratio
        :param p0: float or None
        :param p1: float or None
        :return:
        float, proportions value
        """
        if p1 is None:
            print('Computing p1 from the Odds Ratio')
            return (odds_ratio * p0) / ((1 - p0) + (odds_ratio * p0))
        if p0 is None:
            print('Computing p0 from the Odds Ratio')
            return p1 / (1 - p1) / (odds_ratio + p1 / (1 - p1))

    def get_proportions_from_risk(self,risk_ratio: float, p0: Optional[float] = None,
                                  p1: Optional[float] = None) -> float:
        """
        Get a proportion from the risk ratio
        :param risk_ratio:
        :param p0:
        :param p1:
        :return:
        """
        if p1 is None:
            print('Computing p1 from the Risk Ratio')
            return risk_ratio * p0
        if p0 is None:
            print('Computing p0 from the Risk Ratio')
            return p1/risk_ratio
    @staticmethod
    def calculate_z_values(alpha: float, comparison: str) -> float:
        """

        :param alpha: alpha or beta parameter
        :param comparison: method od comparison
        :return:
            float, z value
        """

        def two_sided_z_value(alpha: float) -> float:
            # Divide alpha by 2 for a two-sided test
            alpha /= 2
            # Calculate z-value
            return norm.ppf(1 - alpha)

        def left_sided_z_value(alpha: float) -> float:
            # For left-sided test, use alpha directly
            return norm.ppf(alpha)

        def right_sided_z_value(alpha: float) -> float:
            # For right-sided test, use (1 - alpha)
            return norm.ppf(1 - alpha)

        if comparison not in ['two sided', 'left sided', 'right sided']:
            raise ValueError(f'Wrong comparison method for the z calculation {comparison}')
        if comparison == 'two sided':
            return two_sided_z_value(alpha=alpha)
        if comparison == 'left sided':
            return left_sided_z_value(alpha=alpha)
        if comparison == 'right sided':
            return right_sided_z_value(alpha=alpha)


if __name__ == "__main__":
    input = ['dichotomous', 'continuous']
    output = ['dichotomous', 'continuous']
    sample_estimate = SampleSizeEstimation(
        feature_type=input[0],
        output_type=output[0],
        alpha=0.05,
        beta=0.20,
        comparison='two sided'
    )
    sample_size_dc = sample_estimate.in_dichotomous_out_continuous(
        effect=0.2,
        std_dev=1.0,
        r=1
    )

    sample_size_dd = sample_estimate.in_dichotomous_out_dichotomous(r=1, effect=0.1, p1=0.3, p0=0.2)

    power_dd = sample_estimate.get_power_from_sample_size(
        difference='proportions',
        r=2,
        p0=0.25,
        sample_size=175,
        odds_ratio=1.8
    )
    sample_estimate.calculate_z_values(alpha=0.20, comparison='two sided')

    # %% Question 1
    print("# %% Question 1")
    sample_estimate = SampleSizeEstimation(
        feature_type=input[0],
        output_type=output[0],
        alpha=0.05,
        beta=0.1,
        comparison='two sided'
    )
    p1 = 4.4 / 10000
    p0 = 2.2 / 10000
    print("Question 1")
    sample_estimate.in_dichotomous_out_dichotomous(
        r=1,
        effect=p1 - p0,
        p1=p1,
        p0=p0,
    )

    # %% Question 2
    print("# %% Question 2")
    print("Question 2")
    sample_estimate.in_dichotomous_out_dichotomous(
        r=1,
        p1=0.25,
        study_type='case-control',
        risk_ratio=2,
    )
    # %% Question 3
    print("Question 3")
    sample_estimate.in_dichotomous_out_dichotomous(
        r=1,
        p1=0.10,
        study_type='case-control',
        risk_ratio=2,
    )
    # %% Question 4
    print("Question 4")
    sample_estimate.get_power_from_sample_size(
        difference='proportions',
        sample_size=400,
        r=1,
        p1=0.25,
        risk_ratio=2,
    )

    # %% Question 5
    print("Question 5")
    sample_estimate.get_power_from_sample_size(
        difference='proportions',
        sample_size=200,
        r=3,
        p1=0.25,
        risk_ratio=2,
    )

odds_male_cases = 300 / 600
odds_male_controls = 100 / 300

# Calculate the odds ratio for males
odds_ratio_males = odds_male_cases / odds_male_controls

# Round the result to the nearest tenth
rounded_odds_ratio_males = round(odds_ratio_males, 1)


# Calculate the odds ratio for males
odds_ratio_males = (300 / 600) / (100 / 300)

# Calculate the odds ratio for females
odds_ratio_females = (4 / 40) / (50 / 400)

# Calculate the Mantel-Haenszel sex-adjusted odds ratio
total_cases_males = 300 + 100
total_controls_males = 900 + 400
total_cases_females = 4 + 50
total_controls_females = 44 + 450

weighted_odds_ratio_males = odds_ratio_males * (total_cases_males / (total_cases_males + total_controls_males))
weighted_odds_ratio_females = odds_ratio_females * (total_cases_females / (total_cases_females + total_controls_females))

adjusted_odds_ratio = weighted_odds_ratio_males + weighted_odds_ratio_females

# Round the result to the nearest hundredth
rounded_adjusted_odds_ratio = round(adjusted_odds_ratio, 2)

print(f"The Mantel-Haenszel sex-adjusted odds ratio for the injury-arthritis association is approximately {rounded_adjusted_odds_ratio}")
