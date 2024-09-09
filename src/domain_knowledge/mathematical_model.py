from scipy.integrate import solve_ivp

import numpy as np
import math
import pandas as pd
from lmfit import minimize, Parameters
from typing import Dict, Tuple, List, Callable

from src.tools.enums import ClassName, ParameterEstimation


# Mathematical model class
class MathematicalModel:
    """
    A class for simulating and analyzing mathematical models for different types of trajectories.

    This class provides functionalities for simulating the motion of objects based on various mathematical models, estimating model parameters, and comparing simulated trajectories with observed data.

    Attributes:
        model_type (ClassName): The type of model (e.g., FISH, PLASTIC).
        input_type (ParameterEstimation): The type of input used for the model.
        sim_traj (pd.DataFrame): The simulated trajectory data.
        param_bounds (dict): Parameter bounds for the model.
        g (float): Acceleration due to gravity.
        rho (float): Density of the fluid.
        states (list): List of state names in the model.
        init_states (list): Initial state values for the simulation.
        t (np.ndarray): Array of time points for the simulation.
        config_data (dict): Configuration data for the model.
        default_params (dict): Default parameters for the model.
        extracted_param (dict): Extracted parameters after fitting.
        n_decimals (int): Number of decimal places for precision.
        params (dict): Parameters used in the simulation.
        equations (callable): The function representing the model's equations.
        inputs (callable): The function representing the model's inputs.

    Methods:
        resample: Resamples the DataFrame at specified time points.
        residuals: Calculates residuals between model output and observed data for parameter fitting.
        add_fitting_params: Adds fitting parameters to the lmfit Parameters object.
        parameter_estimation: Estimates model parameters using the lmfit library.
        compare_extracted_to_estimated: Compares estimated parameters with extracted values.
        unpack_parameters: Unpacks parameters from configuration data.
        equations_pb: Defines the equations for the plastic bottle model.
        equations_f: Defines the equations for the fish model.
        inputs_pb: Defines the inputs for the plastic bottle model.
        inputs_f: Defines the inputs for the fish model.
        solve: Solves the model equations using numerical integration.
    """

    def __init__(
        self,
        config: dict,
        init_states: dict,
        model_type: ClassName,
        tend: float,
        input_type: ParameterEstimation,
        sim_traj: pd.DataFrame = None,
        param_bounds: dict = {},
        tstart: float = 0,
        tsample: float = 0.01,
        n_decimals: int = 10,
    ) -> None:
        """
        Initializes the MathematicalModel with configuration settings, initial states, and model type.

        Parameters:
            config (dict): Configuration settings for the model.
            init_states (dict): Initial state values for the model simulation.
            model_type (ClassName): The type of model to simulate.
            tend (float): The end time for the simulation.
            input_type (ParameterEstimation): Type of input used for the model.
            sim_traj (pd.DataFrame, optional): DataFrame containing simulated trajectory data.
            param_bounds (dict, optional): Parameter bounds for the model.
            tstart (float, optional): The start time for the simulation. Default is 0.
            tsample (float, optional): The time step for the simulation. Default is 0.01.
            n_decimals (int, optional): Number of decimal places for precision. Default is 10.
        """
        # Preprocess values
        self.n_decimals = n_decimals
        self.model_type = model_type
        self.input_type = input_type
        self.sim_traj = sim_traj

        self.param_bounds = param_bounds

        self.g = 9.800000190734864
        self.rho = 997

        self.states = ["y", "vy", "z", "vz"]

        self.init_states = [
            init_states[self.states[0] + "_0"],
            init_states[self.states[1] + "_0"],
            init_states[self.states[2] + "_0"],
            init_states[self.states[3] + "_0"],
        ]

        self.t = np.linspace(tstart, tend, int((tend - tstart) / tsample))

        self.config_data = config

        self.first = True

        all_param = ["cd_y", "cd_z", "v_current_z", "V", "m", "A_z", "A_y", "F_fz"]

        default_params = {
            "cd_y": config["cd_y"],
            "cd_z": config["cd_z"],
            "v_current_z": config["waterCurrentStrength"][0]["z"],
        }

        default_params_keys = list(default_params.keys())

        self.params = {}

        if self.model_type == ClassName.FISH:
            self.equations = self.equations_f
            self.inputs = self.inputs_f
        elif self.model_type == ClassName.PLASTIC:
            self.equations = self.equations_pb
            self.inputs = self.inputs_pb
        else:
            raise ValueError(f"Input model type not recognized: {self.model_type}")

        if self.input_type != ParameterEstimation.OFF:

            if self.input_type == ParameterEstimation.FULL:
                param_to_extract = []

            if self.input_type == ParameterEstimation.HIGH:
                param_to_extract = ["F_fz"]

            if self.input_type == ParameterEstimation.MEDIUM:
                param_to_extract = ["A_z", "A_y", "F_fz"]

            if self.input_type == ParameterEstimation.LOW:
                param_to_extract = ["A_z", "A_y", "F_fz", "m"]

            self.param_to_estimate = [
                s for s in all_param if s not in default_params_keys + param_to_extract
            ]

            self.df_sim_traj = self.resample(self.sim_traj)

            self.params = default_params.copy()
            extracted_param = self.unpack_parameters(param_to_extract)
            self.params.update(extracted_param)

            estimated_param = self.parameter_estimation(self.param_to_estimate)
            self.params.update(estimated_param)
        elif self.input_type == ParameterEstimation.OFF:
            self.params = self.unpack_parameters(all_param)

        else:
            raise NotImplementedError(
                f"The mathematical model input {self.input_type} is not implemented yet."
            )

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples the DataFrame at specified time points and interpolates missing data.

        Parameters:
            df (pd.DataFrame): The dataframe to be resampled.

        Returns:
            pd.DataFrame: The resampled and interpolated dataframe.
        """

        if "t" in df.columns:
            ts = df["t"].tolist()
            df.set_index("t", inplace=True)

        else:
            ts = df.index.tolist()

        pd.set_option("display.max_rows", None)

        ts = [item for item in ts if item not in self.t]
        common_index = df.index.union(self.t)

        # Reindex and interpolate the dataframes
        df = df.reindex(common_index).interpolate(method="linear")

        # Then interpolate backward for any remaining NaNs
        df.interpolate(method="linear", limit_direction="backward", inplace=True)
        # Drop 'x' and 'vx' columns if they exist
        if "x" in df.columns:
            df = df.drop("x", axis=1)
        if "vx" in df.columns:
            df = df.drop("vx", axis=1)

        df_filtered = df.drop(ts)

        return df_filtered

    def residuals(
        self, fitting_params: Parameters, inputs, observed_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculates the residuals between the mathematical model output and observed data for parameter estimation.

        Parameters:
            fitting_params (Parameters): The lmfit Parameters object for the fitting process.
            inputs: The input function for the model.
            observed_data (pd.DataFrame): The observed data for comparison.

        Returns:
            np.ndarray: An array of residuals.
        """
        # Extract variable parameters from lmfit's parameter structure

        new_values = {}

        for param in self.param_to_estimate:
            if param == "F_fz":
                if self.model_type == ClassName.FISH:
                    new_values[param] = fitting_params[param].value
                else:
                    continue
            else:
                new_values[param] = fitting_params[param].value

        self.params.update(new_values)

        df = self.solve()
        df_reindex = df.set_index("t")

        residual = observed_data - df_reindex

        return residual.values.ravel()

    def add_fitting_params(self, obj: Parameters, key: str) -> None:
        """
        Adds a fitting parameter to the lmfit Parameters object.

        Parameters:
            obj (Parameters): The lmfit Parameters object.
            key (str): The key of the parameter to be added.
        """
        obj.add(
            key,
            value=self.param_bounds[key]["mid"],
            min=self.param_bounds[key]["min"],
            max=self.param_bounds[key]["max"],
        )

    def parameter_estimation(self, param_to_estimate: list) -> Dict:
        """
        Runs the parameter estimation process to fit the model to the observed data.

        Parameters:
            param_to_estimate (list): The level of parameter estimation, options are low, medium and high

        Returns:
            Dict: A dictionary containing the estimated parameter values.
        """

        fitting_params = Parameters()
        for param in param_to_estimate:
            if param == "F_fz":
                if self.model_type == ClassName.FISH:
                    self.add_fitting_params(fitting_params, "F_fz")
                else:
                    continue
            else:
                self.add_fitting_params(fitting_params, param)

        out = minimize(
            self.residuals,
            fitting_params,
            args=(self.inputs, self.df_sim_traj),
            ftol=1e-8,  # desired value for ftol
        )

        output_dict = {}
        for param in param_to_estimate:
            if param == "F_fz":
                if self.model_type == ClassName.FISH:
                    output_dict[param] = out.params[param].value
                else:
                    continue
            else:
                output_dict[param] = out.params[param].value

        extr_dict = {
            "m": self.config_data["m"],
            "V": self.config_data["V"],
            "F_fz": self.config_data["swimForceVector"]["z"],
        }

        # (
        #     _,
        #     _,
        #     parameter_estimation_comparison,
        # ) = self.compare_extracted_to_estimated(estm=output_dict, truth=extr_dict)
        self.parameter_estimation_dict = {
            # "truth": extr_dict,
            "estm": output_dict,
            # "comparison": parameter_estimation_comparison,
        }

        return output_dict

    def compare_extracted_to_estimated(
        self, estm: Dict, truth: Dict
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Compares estimated parameter values with the extracted or true values.

        Parameters:
            estm (Dict): The estimated parameter values.
            truth (Dict): The extracted or true parameter values.

        Returns:
            Tuple[Dict, Dict, Dict]: A tuple containing the estimated values, true values, and a comparison dictionary.
        """

        output_dict = {}

        for ky in estm.keys():
            mean_squared_error = np.sqrt((truth[ky] - estm[ky]) ** 2)
            # norm = mean_squared_error / abs(truth[ky])
            norm = mean_squared_error
            output_dict[ky] = (1 - norm) * 100

        return estm, truth, output_dict

    def unpack_parameters(self, param_to_unpack) -> Dict:
        """
        Unpacks parameters from the model's configuration data.
        Parameters:
            param_to_unpack (list): Parameters that should be unpacked from the list

        Returns:
            unpacked_param (dict): A dictionary containing unpacked parameter values.
        """
        unpacked_param = {}
        data_source = self.config_data

        for param in param_to_unpack:
            if param == "F_fz":

                F_fz = round(data_source["swimForceVector"]["z"], self.n_decimals)
                if self.model_type == ClassName.PLASTIC:
                    F_fz = 0

                if F_fz == 0 and self.model_type == ClassName.FISH:
                    F_fz = 7

                unpacked_value = F_fz

            elif param == "v_current_z":
                unpacked_value = round(
                    data_source["waterCurrentStrength"][0]["z"], self.n_decimals
                )
            else:
                unpacked_value = round(data_source[param], self.n_decimals)

            unpacked_param[param] = unpacked_value

        return unpacked_param

    def equations_pb(
        self,
        states: List[float],
        t: float,
        params: Dict[str, float],
        inputs: Callable[[float], Tuple[float, float, float]],
    ) -> List[float]:
        """
        Defines the equations for the plastic bottle model.

        Parameters:
            states (List[float]): The current states of the model.
            t (float): The current time.
            params (Dict[str, float]): The parameters of the model.
            inputs (Callable[[float], Tuple[float, float, float]]): A function that provides the inputs for the model at a given time.

        Returns:
            List[float]: The derivatives of the states as a list.
        """

        x1, x2, x3, x4 = states

        m = params["m"]
        V = params["V"]
        A_y = params["A_y"]
        A_z = params["A_z"]
        cd_y = params["cd_y"]
        cd_z = params["cd_z"]
        v_current_z = params["v_current_z"]
        u1, u2, u3 = inputs(t)

        dxdt = [
            x2,
            1
            / m
            * (
                -(m * self.g)
                + (self.rho * self.g * V)
                - (np.sign(x2) * 0.5 * cd_y * self.rho * A_y * (x2) ** 2)
            ),
            x4,
            1
            / m
            * (
                -(
                    np.sign(x4 - v_current_z)
                    * 0.5
                    * cd_z
                    * self.rho
                    * A_z
                    * (x4 - v_current_z) ** 2
                )
            ),
        ]

        return dxdt

    def equations_f(
        self,
        states: List[float],
        t: float,
        params: Dict[str, float],
        inputs: Callable[[float], Tuple[float, float, float]],
    ) -> List[float]:
        """
        Defines the equations for the fish model.

        Parameters:
            states (List[float]): The current states of the model.
            t (float): The current time.
            params (Dict[str, float]): The parameters of the model.
            inputs (Callable[[float], Tuple[float, float, float]]): A function that provides the inputs for the model at a given time.

        Returns:
            List[float]: The derivatives of the states as a list.
        """
        x1, x2, x3, x4 = states
        m = params["m"]
        V = params["V"]
        A_y = params["A_y"]
        A_z = params["A_z"]
        cd_y = params["cd_y"]
        cd_z = params["cd_z"]
        v_current_z = params["v_current_z"]
        F_fz = params["F_fz"]
        u1, u2, u3 = inputs(t)

        # F_fz = 7

        dxdt = [
            x2,
            1
            / m
            * (
                -(m * self.g)
                + (self.rho * self.g * V)
                - (np.sign(x2) * 0.5 * cd_y * self.rho * A_y * (x2) ** 2)
            ),
            x4,
            1
            / m
            * (
                F_fz * u2
                + F_fz
                - (
                    np.sign(x4 - v_current_z)
                    * 0.5
                    * cd_z
                    * self.rho
                    * A_z
                    * (x4 - v_current_z) ** 2
                )
            ),
        ]

        return dxdt

    def inputs_pb(self, t: float) -> Tuple:
        """
        Defines the inputs for the plastic bottle model.

        Parameters:
            t (float): The current time.

        Returns:
            Tuple: The input values for the model at the given time.
        """

        u1 = 0
        u2 = 0
        u3 = 0

        return u1, u2, u3

    def inputs_f(self, t: float) -> Tuple:
        """
        Defines the inputs for the fish model.

        Parameters:
            t (float): The current time.

        Returns:
            Tuple: The input values for the model at the given time.
        """

        u1 = 0
        u2 = math.sin(t * 50)
        u3 = 0

        return u1, u2, u3

    def solve(self) -> pd.DataFrame:
        """
        Solves the mathematical model equations using numerical integration.

        Returns:
            pd.DataFrame: The DataFrame containing the solution of the model.
        """

        successful_method = None
        methods = ["BDF"]
        for method in methods:
            solution = solve_ivp(
                lambda t, y: self.equations(y, t, self.params, self.inputs),
                [self.t[0], self.t[-1]],
                self.init_states,
                t_eval=self.t,
                method=method,
                max_step=0.1,
            )

            if solution.success:
                # print(
                #     "The solver successfully reached the end of the integration interval using",
                #     method,
                # )
                successful_method = method
                break
            else:
                print(
                    f"The solver {method} could not reach the end of the integration interval. Trying next method..."
                )

        # Final message regarding the overall success or failure
        if successful_method:
            # print(f"The system was successfully solved using {successful_method}.")
            pass
        else:
            print("All methods failed to solve the system.")

        # Storing the solution regardless of success to inspect the results
        self.df = pd.DataFrame(
            solution.y.T, columns=self.states
        )  # solution.y is transposed to align with column names
        self.df["t"] = solution.t

        # Clipping values after integration
        self.df["y"] = np.clip(self.df["y"], -4, 4)
        self.df["z"] = np.clip(self.df["z"], -4, 4)
        self.df["vy"] = np.clip(self.df["vy"], -4, 4)
        self.df["vz"] = np.clip(self.df["vz"], -4, 4)
        return self.df
