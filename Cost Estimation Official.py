# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 13:42:46 2025

@author: Katie
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import copy

# ======================== Recurring =================================
def estimate_structure(inputs):
    # Mass split (in kg)
    mass_composite = 7.5
    mass_aluminum = 30.0

    # Material cost ($/kg)
    material_costs = {
        'composite': 2.68,
        'aluminum': 5.16
    }

    # Manufacturing cost ($/kg)
    manufacturing_costs = {
        'composite': 350,
        'aluminum': 150
    }

    material_cost = (
        mass_composite * material_costs['composite'] +
        mass_aluminum * material_costs['aluminum']
    )

    manufacturing_cost = (
        mass_composite * manufacturing_costs['composite'] +
        mass_aluminum * manufacturing_costs['aluminum']
    )

    return material_cost, manufacturing_cost

def estimate_airframe(inputs):
    material_cost, manufacturing_cost = estimate_structure(inputs)

    # Apply deploy mechanism multiplier to manufacturing only
    if inputs.get('DeployMechanism'):
        manufacturing_cost *= 1.2  # 20% more expensive to manufacture

    return {
        'materials': material_cost,
        'manufacturing': manufacturing_cost,
    }
def estimate_landing_gear(inputs):
    num_legs = 4
    leg_mass_kg = 1.2
    material_cost_per_kg = 5.16
    wheel_unit_cost = 750
    integration_multiplier = 1.10

    structure_material = num_legs * leg_mass_kg * material_cost_per_kg
    wheels = 2 * wheel_unit_cost

    base_total = structure_material + wheels
    gear_cost = base_total * integration_multiplier

    return {
        'materials': structure_material,
        'cots': wheels,
        'manufacturing': (gear_cost - structure_material - wheels)
    }


def estimate_propulsion(inputs):
    # Base prices per rotor
    base_motor_cost_per_rotor = 2187.5
    base_prop_cost_per_rotor = 800

    # Quantities
    num_motors = inputs['Motors']
    num_props = inputs['Propellers']

    # Base COTS costs
    motor_cost = num_motors * base_motor_cost_per_rotor
    propeller_base_cost = num_props * base_prop_cost_per_rotor

    # Split rotors: 50% normal, 50% CycloRotors (COTS)
    normal_motor_cost = 0.5 * motor_cost
    normal_prop_cost = 0.5 * propeller_base_cost

    cyclo_motor_cost = 0.5 * motor_cost
    cyclo_prop_cost = 0.5 * propeller_base_cost

    # Apply contingency uplift for CycloRotors (30% more expensive)
    cyclo_uplift = 1.3 
    cyclo_total = (cyclo_motor_cost + cyclo_prop_cost) * cyclo_uplift

    # Manufacturing uplift only for serrated props on normal rotors
    propeller_manuf_cost = 0
    if inputs.get('SerratedPropellers'):
        serration_uplift = 0.25  # 25% added mfg cost (contingency)
        propeller_manuf_cost = normal_prop_cost * serration_uplift

    total_cots = normal_motor_cost + normal_prop_cost + cyclo_total
    total_manuf = propeller_manuf_cost  # Only serration adds mfg cost

    return {
        'cots': total_cots,
        'manufacturing': total_manuf
    }


def estimate_energy(inputs): 
    # Owl‑22 battery size
    capacity_kwh = 22.9

    # Solid-state battery cost (2025 estimate)
    current_cost_per_kwh = 450  # USD/kWh, includes baseline pack cost

    # Aviation integration premium (packaging, BMS, certification)
    premium = 1.2

    # Future cost decline (assume 10% per year for 5 years)
    years_ahead = 5
    annual_decline = 0.10
    discount_factor = (1 - annual_decline) ** years_ahead  # ~0.59049

    # Final cost
    future_cost_per_kwh = current_cost_per_kwh * discount_factor
    battery_pack_cost = capacity_kwh * future_cost_per_kwh * premium
    return battery_pack_cost

def estimate_avionics(inputs):
    # Updated unit prices based on eVTOL industry ranges (certified hardware)
    gnss = 250  # high-accuracy unit
    obc = 600   # onboard computer w/ IMU
    lidar_ground = 500
    lidar_obstacles = 350 * 5  # high-spec LiDARs
    camera = 250

    # Flight control & safety systems
    flight_controller = 3000
    redundancy_hardware = 1500
    telemetry_hardware = 1000

    base_cost = (
        gnss + obc + lidar_ground + lidar_obstacles + camera +
        flight_controller + redundancy_hardware + telemetry_hardware
    )

    # Contingency for integration, flight safety, shielding, EMI compliance
    contingency_multiplier = 1.2

    total_avionics_cost = base_cost * contingency_multiplier
    return total_avionics_cost


def estimate_wiring(inputs):
    return 3000 #estimation based on Volocity


def estimate_cooling(inputs): 
    return 5000 #estimation based on Volocity

def estimate_recurring(inputs):
    return {
        'Airframe': estimate_airframe(inputs), 
        'Landing Gear': estimate_landing_gear(inputs),
        'Propulsion': estimate_propulsion(inputs),
        'Battery': estimate_energy(inputs),
        'Avionics': estimate_avionics(inputs),
        'Electronics': estimate_wiring(inputs),
        'AC System': estimate_cooling(inputs),
    }


# 1. Categorize recurring costs into COTS, manufacturing
def categorize_recurring_costs(inputs):
    all_components = estimate_recurring(inputs)

    airframe = all_components['Airframe']
    landing = all_components['Landing Gear']
    propulsion = all_components['Propulsion']

    # COTS baseline
    cots_total = (
        all_components['Battery'] +
        all_components['Avionics'] +
        all_components['Electronics'] +
        all_components['AC System'] +
        landing['cots'] +
        propulsion['cots']
    )

    manufacturing_total = (
        airframe['manufacturing'] +
        landing['manufacturing'] +
        propulsion['manufacturing']
    )

    material_total = (
        airframe['materials'] +
        landing['materials']
    )

    # Add raw material to COTS
    cots_total += material_total

    # 🔧 Apply contingency for omitted recurring elements
    cots_contingency_rate = 0.10  # 10% for items like interior, signage, mounts
    cots_total *= (1 + cots_contingency_rate)

    return cots_total, manufacturing_total



# 2. Plot cost breakdown vs production volume
def plot_cost_vs_production(inputs):
    cots_base, manuf_base = categorize_recurring_costs(inputs)

    # Batch sizes (smoother resolution for better visual)
    N = np.linspace(1, 1500, 100)

    # --- NASA-style COTS reduction (log-based) ---
    ln_n = np.log(N)
    discount_factor = 1 - (0.0721 * ln_n - 0.2322)
    discount_factor = np.clip(discount_factor, 0, 1)
    cots = cots_base * discount_factor
    
    # --- NASA-style Manufacturing learning curve ---
    learning_rate = 0.85
    learning_exponent = np.log10(learning_rate) / np.log10(2)
    manuf = manuf_base * N**learning_exponent

    # Assembly and total cost
    base_cost = cots + manuf
    assembly = 0.4 * base_cost
    total = base_cost + assembly

    # Plot as stacked area
    plt.figure(figsize=(9, 5))
    plt.stackplot(N, cots, manuf, assembly,
                  labels=["COTS", "Manufacturing", "Assembly"],
                  colors=["#6baed6", "#f28e2b", "#bdbdbd"])


    # Create interpolators
    interp_total = interp1d(N, total)
    interp_cots = interp1d(N, cots)
    interp_manuf = interp1d(N, manuf)
    interp_assembly = interp1d(N, assembly)
    
    # Get value exactly at N = 1000
    n_target = 1000
    t_val = interp_total(n_target)
    c_val = interp_cots(n_target)
    m_val = interp_manuf(n_target)
    a_val = interp_assembly(n_target)

    # Vertical lines at batch size = 1 and 1000
    idx_1 = np.argmin(np.abs(N - 1))
    
    plt.axvline(x=1000, color='teal', linewidth=2)
    plt.text(1000, t_val + 1000, f"{t_val:,.0f}$", fontsize=10, fontweight='bold')

    plt.axvline(x=1, color='teal', linewidth=2)
    plt.text(1, total[idx_1] + 1000, f"{total[idx_1]:,.0f}$", fontsize=10, fontweight='bold')


    # Final labels and styling
    plt.xlabel("Produced batch size [#]")
    plt.ylabel("unit production cost [$]")
    plt.title("Owl-22 Unit Cost Breakdown vs Batch Size")
    plt.legend(loc="upper right")
    plt.xlim([1, 1500])
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", linewidth=0.5, axis="y")
    plt.tight_layout()
    plt.show()

    print(f"Estimated unit production cost at 1000 units: ${t_val:,.2f}")
    print(f"  - COTS: ${c_val:,.2f}")
    print(f"  - Manufacturing: ${m_val:,.2f}")
    print(f"  - Assembly: ${a_val:,.2f}")


# ============================= Non-recurring ==========================
def estimate_rnd(inputs):
   # Got info from Joby
    base_rnd = 450_000_000  
    software_addition = 2_000_000 if inputs.get('AutomationLevel') == 'semi' else 1_000_000
    return base_rnd + software_addition

def estimate_certification(inputs):
    # FAA wiki: ~$1M for < 3 seat certs, $25M for GA aircraft 
    base = 1_000_000 if inputs.get('Passengers', 1) <= 3 else 25_000_000
    if inputs.get('AutomationLevel') == 'semi':
        base *= 1.5
    return base

def estimate_facilities(inputs):
    # Assumed facility size: 2,000 m² in UAE
    construction = 4_000_000  # upgraded facility specs
    tooling = 1_200_000
    machinery = 2_000_000
    support_systems = 800_000
    office_infra = 500_000
    it_systems = 400_000
    installation_commissioning = 0.15 * (tooling + machinery)
    land_prep_or_lease = 1_000_000
    training = 300_000
    certification_setup = 300_000

    subtotal = (
        construction + tooling + machinery + support_systems + office_infra +
        it_systems + installation_commissioning + land_prep_or_lease +
        training + certification_setup
    )

    contingency = 0.15 * subtotal  # 15% safety margin from NASA
    return subtotal + contingency


def estimate_patents(inputs):
    return 150_000


def estimate_nonrecurring(inputs):
    return {
        'R&D': estimate_rnd(inputs),
        'Certification': estimate_certification(inputs),
        'Facility Investment': estimate_facilities(inputs),
        'Patent & IP': estimate_patents(inputs),
    }

def estimate_nonrecurring_per_unit(inputs):
    rnd = estimate_rnd(inputs)
    cer = estimate_certification(inputs)
    fac = estimate_facilities(inputs)
    patent = estimate_patents(inputs)
    
    #Investment time frame
    time = 15        #years
    units = time * 1000
    
    #R&D Amortization
    rnd_am = (0.5*rnd) / units
    
    #Certification Amortization
    cer_am = cer / units
    
    #Facilities Depreciation
    fac_dep = fac / units
    
    #Patent Amortization 
    pat_am = patent / units
    
    #Total Fixed Costs per unit
    AmDep = rnd_am + cer_am + fac_dep + pat_am
    
    return AmDep


# === Estimate total costs ===
def estimate_total_cost(inputs):
    recurring = estimate_recurring(inputs)
    nonrecurring = estimate_nonrecurring(inputs)

    recurring_total = 0
    for key, val in recurring.items():
        if isinstance(val, dict):
            recurring_total += sum(val.values())  # sum Airframe & Landing Gear parts
        else:
            recurring_total += val

    nonrecurring_total = sum(nonrecurring.values())

    breakdown = {
        'Recurring': recurring,
        'Non-Recurring': nonrecurring,
        'Recurring Total': recurring_total,
        'Non-Recurring Total': nonrecurring_total,
        'Total': recurring_total + nonrecurring_total
    }

    return breakdown


# ==== Estimate price based on recurring costs ===
def estimate_price_at_volume(inputs, production_volume, profit_margin=0.23):
    cots_base, manuf_base = categorize_recurring_costs(inputs)
    N = np.linspace(1, 1500, 1000)

    # --- NASA-style COTS reduction ---
    ln_n = np.log(N)
    discount_factor = 1 - (0.0721 * ln_n - 0.2322)
    discount_factor = np.clip(discount_factor, 0, 1)
    cots = cots_base * discount_factor
    
    # --- Manufacturing learning curve ---
    learning_rate = 0.85
    learning_exponent = np.log10(learning_rate) / np.log10(2)
    manuf = manuf_base * N**learning_exponent

    base_cost = cots + manuf
    assembly = 0.4 * base_cost
    total = base_cost + assembly

    # Interpolation
    interp_total = interp1d(N, total)
    recurring_unit_cost = float(interp_total(production_volume))

    nonrecurring = estimate_nonrecurring(inputs)
    nonrecurring_total = sum(nonrecurring.values())
    
    nonrecurring_per_unit = estimate_nonrecurring_per_unit(inputs)

    unit_cost = recurring_unit_cost + nonrecurring_per_unit
    unit_price = unit_cost * (1 + profit_margin)

    return unit_price, recurring_unit_cost, nonrecurring_total

# =============== Sensitivity Analysis Recurring Costs ============
def plot_total_cost_vs_contingency(inputs, production_volume=1000, focus='recurring'):
    import matplotlib.pyplot as plt
    import copy

    # Sweep values for each contingency
    sweeps = {
        'avionics_multiplier': [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
        'cots_contingency': [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
        'cyclorotor_uplift': [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
        'serration_uplift': [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
        'deploy_multiplier': [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
    }

    default_values = {
        'avionics_multiplier': 1.2,
        'cots_contingency': 1.10,
        'cyclorotor_uplift': 1.3,
        'serration_uplift': 1.25,
        'deploy_multiplier': 1.2
    }

    def run_model(overrides):
        def custom_estimate_avionics(inputs):
            base = 250 + 600 + 500 + 350*5 + 250 + 3000 + 1500 + 1000
            return base * overrides['avionics_multiplier']

        def custom_estimate_airframe(inputs):
            mat_cost, mfg_cost = estimate_structure(inputs)
            if inputs.get('DeployMechanism'):
                mfg_cost *= overrides['deploy_multiplier']
            return {'materials': mat_cost, 'manufacturing': mfg_cost}

        def custom_estimate_propulsion(inputs):
            base_motor_cost = 2187.5
            base_prop_cost = 800
            n_motors = inputs['Motors']
            n_props = inputs['Propellers']

            motor_cost = n_motors * base_motor_cost
            prop_cost = n_props * base_prop_cost

            normal_motor = 0.5 * motor_cost
            normal_prop = 0.5 * prop_cost
            cyclo_motor = 0.5 * motor_cost
            cyclo_prop = 0.5 * prop_cost

            cyclo_total = (cyclo_motor + cyclo_prop) * overrides['cyclorotor_uplift']
            serration_mfg = 0
            if inputs.get('SerratedPropellers'):
                serration_mfg = normal_prop * overrides['serration_uplift']

            total_cots = normal_motor + normal_prop + cyclo_total
            return {'cots': total_cots, 'manufacturing': serration_mfg}

        def custom_categorize(inputs):
            airframe = custom_estimate_airframe(inputs)
            landing = estimate_landing_gear(inputs)
            propulsion = custom_estimate_propulsion(inputs)

            cots = (
                estimate_energy(inputs) +
                custom_estimate_avionics(inputs) +
                estimate_wiring(inputs) +
                estimate_cooling(inputs) +
                landing['cots'] +
                propulsion['cots'] +
                airframe['materials'] +
                landing['materials']
            )
            manuf = airframe['manufacturing'] + landing['manufacturing'] + propulsion['manufacturing']
            cots *= overrides['cots_contingency']
            return cots, manuf

        # Override just categorize function
        original_categorize = categorize_recurring_costs
        globals()['categorize_recurring_costs'] = custom_categorize

        price, recurring_cost, _ = estimate_price_at_volume(inputs, production_volume, profit_margin=0.23)

        # Restore original function
        globals()['categorize_recurring_costs'] = original_categorize

        return recurring_cost if focus == 'recurring' else price

    # Plot setup
    plt.figure(figsize=(10, 6))

    for param, sweep_values in sweeps.items():
        results = []
        for val in sweep_values:
            # Apply only one parameter change
            overrides = default_values.copy()
            overrides[param] = val
            y = run_model(overrides)
            results.append(y)

        # Adjust x values to show % if additive (contingencies), otherwise raw multipliers
        x_vals = sweep_values

        plt.plot(x_vals, results, marker='o', label=param.replace('_', ' ').title())

    # Formatting
    plt.xlabel("Effective Cost Multiplier")
    ylabel = "Recurring Cost per Unit [$]" if focus == 'recurring' else "Selling Price per Unit [$]"
    plt.ylabel(ylabel)
    plt.title(f"Effect of Individual Contingencies on {ylabel}")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

# ===================== ROI Analysis ===============================

def npv_sensitivity_analysis(selling_price, recurring_unit_cost, nrc_total,
                             units_per_year=1000, production_years=15,
                             discount_rate_range=(0.04, 0.12), steps=50):

    profit_per_unit = selling_price - recurring_unit_cost
    annual_profit = profit_per_unit * units_per_year

    discount_rates = np.linspace(discount_rate_range[0], discount_rate_range[1], steps)
    npvs = []

    for r in discount_rates:
        years = np.arange(6, 6 + production_years)  # Cash flow starts in year 6
        discount_factors = 1 / (1 + r) ** years
        discounted_cash_flows = annual_profit * discount_factors
        npv = discounted_cash_flows.sum() - nrc_total
        npvs.append(npv)

    avg_npv = np.mean(npvs)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(discount_rates * 100, npvs, linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("NPV Sensitivity to Discount Rate")
    plt.xlabel("Discount Rate [%]")
    plt.ylabel("Net Present Value [USD]")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return discount_rates, npvs, avg_npv

def calculate_simple_roi(selling_price, recurring_unit_cost, nrc_total,
                         total_units=15000):

    # Per-unit profit
    profit_per_unit = selling_price - recurring_unit_cost
    
    # Total profit across all units
    total_profit = profit_per_unit * total_units

    # Simple ROI
    roi = total_profit / nrc_total
    return roi


# =================== Owl-22 Configuration =============================

owl_22_specs = {
    'MTOW': 300,
    'OW': 200,
    'Range_km': 30,
    'Speed_kmh': 60,
    'Passengers': 1,
    'Motors': 8,
    'Propellers': 8,
    'SerratedPropellers': True,
    'DeployMechanism': True,
    'AutomationLevel': 'semi'
}

# === Run Model ===
if __name__ == "__main__":

    cost_breakdown = estimate_total_cost(owl_22_specs)
    plot_cost_vs_production(owl_22_specs) 

    # Estimate price per unit at batch size 1000
    unit_price, recurring_per_unit, nonrecurring_total = estimate_price_at_volume(
        owl_22_specs, production_volume=1000
    )
    
    # To show how recurring cost changes:
    plot_total_cost_vs_contingency(owl_22_specs, focus='recurring')

    # Or to show how final selling price changes:
    # plot_total_cost_vs_contingency(owl_22_specs, focus='price')

    # Run your cost model as usual
    cost_breakdown = estimate_total_cost(owl_22_specs)

    selling_price = 99_575.09
    recurring_unit_cost = 64_936.89
    nrc_total = cost_breakdown['Non-Recurring Total']
    total_units = 15_000

    discount_rates, npvs, avg_npv = npv_sensitivity_analysis(
        selling_price=selling_price,
        recurring_unit_cost=recurring_unit_cost,
        nrc_total=nrc_total
    )

    # Calculate ROI
    roi = calculate_simple_roi(
        selling_price=selling_price,
        recurring_unit_cost=recurring_unit_cost,
        nrc_total=nrc_total,
        total_units=total_units
        )

    print(f"Simple ROI over the production lifetime: {roi:.2f}x ({roi*100:.2f}%)")


    print(f"\n=== NPV Sensitivity Analysis ===")
    print(f"Average NPV across 4%–12% discount rates: ${avg_npv:,.2f}")



    print("\n=== Estimated Selling Price at 1000 Units ===")
    print(f"  - Total Price per Unit (incl. profit): ${unit_price:,.2f}")
    print(f"  - Recurring Cost per Unit: ${recurring_per_unit:,.2f}")
    print(f"  _ Total Non-Recurring Cost: ${nonrecurring_total:,.2f}")

    print("\n=== Cost Breakdown for Owl-22 ===")
    for category, items in cost_breakdown.items():
        if isinstance(items, dict):
            print(f"\n-- {category} --")
            for sub, val in items.items():
                if isinstance(val, dict):
                    print(f"{sub:30}:")
                    for subkey, subval in val.items():
                        print(f"  {subkey:28}: ${subval:,.2f}")
                else:
                    print(f"{sub:30}: ${val:,.2f}")
        else:
            print(f"{category:30}: ${items:,.2f}")





