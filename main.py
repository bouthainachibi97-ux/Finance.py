from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

app = FastAPI(
    title="FP&A Advanced Financial Engine API",
    description="API لتشغيل المحرك المالي المتقدم وتوقع السيولة والأرباح",
    version="1.0.0"
)

# ---------------------------------------------------------
# 1. تعريف نماذج البيانات المدخلة والمخرجة (Pydantic Models)
# ---------------------------------------------------------

class FPAInputs(BaseModel):
    sales_units: List[float] = Field(..., description="قائمة بعدد الوحدات المتوقع بيعها شهرياً")
    price_per_unit: float = Field(..., description="سعر بيع الوحدة الأساسي")
    var_cost_per_unit: float = Field(..., description="التكلفة المتغيرة الأساسية للوحدة")
    fixed_costs: float = Field(..., description="التكاليف الثابتة الشهريّة الأساسية")
    credit_terms: List[float] = Field(..., description="شروط الائتمان [نقدي, بعد 30 يوم, بعد 60 يوم]")
    initial_cash: float = Field(..., description="السيولة النقدية البدائية")
    cogs_payment_delay: Optional[int] = Field(1, description="تأخير سداد الموردين بالأشهر")
    
    # المعلمات المتقدمة
    annual_inflation_rate: Optional[float] = Field(0.05, description="معدل التضخم السنوي المتوقع")
    fx_rate_change: Optional[float] = Field(0.0, description="نسبة تغير سعر الصرف للمواد الخام المستوردة")
    fx_cogs_weight: Optional[float] = Field(0.40, description="نسبة المادة الخام المستوردة من التكلفة المتغيرة")
    bad_debt_rate: Optional[float] = Field(0.03, description="نسبة الديون المعدومة غير المقبوضة")
    
    # خطة شراء الأصول: {الشهر: [التكلفة, العمر_بالسنوات]}
    # نستخدم Dict[str, List[float]] لأن مفاتيح JSON تكون دائماً نصية
    capex_plan: Optional[Dict[str, List[float]]] = Field(None, description="خطة الاستثمار الرأسمالي {الشهر: [التكلفة, العمر]}")
    
    tax_rate: Optional[float] = Field(0.25, description="نسبة ضريبة أرباح الشركات")
    tax_payment_months: Optional[List[int]] = Field([3, 9], description="الأشهر التي تُسدد فيها الضريبة (0-indexed)")
    min_safety_cash: Optional[float] = Field(5000, description="حد الأمان النقدي للتنبيهات")

# ---------------------------------------------------------
# 2. نقطة الاتصال الرئيسية (API Endpoint)
# ---------------------------------------------------------

@app.post("/run-fpa", summary="تشغيل التنبؤ المالي")
def run_fpa_endpoint(data: FPAInputs):
    months = len(data.sales_units)
    monthly_inflation = (1 + data.annual_inflation_rate) ** (1/12) - 1
    
    # تهيئة جداول الحسابات
    revenue = np.zeros(months)
    variable_costs = np.zeros(months)
    fixed_costs_adjusted = np.zeros(months)
    depreciation = np.zeros(months)
    capex_outflows = np.zeros(months)
    
    # 1. حساب التعديلات الهيكلية
    for t in range(months):
        inflation_factor = (1 + monthly_inflation) ** t
        fx_cost_factor = 1 + (data.fx_rate_change * data.fx_cogs_weight)
        
        adj_price = data.price_per_unit * inflation_factor
        adj_var_cost = data.var_cost_per_unit * inflation_factor * fx_cost_factor
        
        revenue[t] = data.sales_units[t] * adj_price
        variable_costs[t] = data.sales_units[t] * adj_var_cost
        fixed_costs_adjusted[t] = data.fixed_costs * inflation_factor

    # 2. معالجة الاستثمار الرأسمالي (CAPEX & Depreciation)
    active_assets = []
    if data.capex_plan:
        for m_str, asset_info in data.capex_plan.items():
            m = int(m_str)
            if m < months:
                cost, useful_life_years = asset_info[0], asset_info[1]
                capex_outflows[m] = cost
                monthly_depr = cost / (useful_life_years * 12)
                active_assets.append({'start_month': m, 'monthly_depr': monthly_depr, 'total_months': useful_life_years * 12})
    
    for t in range(months):
        for asset in active_assets:
            if t >= asset['start_month'] and (t - asset['start_month']) < asset['total_months']:
                depreciation[t] += asset['monthly_depr']

    # 3. حساب الأرباح والمصاريف المحاسبية (Accrual P&L)
    ebitda = revenue - (variable_costs + fixed_costs_adjusted)
    ebit = ebitda - depreciation
    
    bad_debt_expense = revenue * (1 - data.credit_terms[0]) * data.bad_debt_rate
    ebt = ebit - bad_debt_expense
    
    tax_expense = np.maximum(0, ebt) * data.tax_rate
    tax_payments = np.zeros(months)
    
    accrued_tax = 0
    for t in range(months):
        accrued_tax += tax_expense[t]
        if t in data.tax_payment_months:
            tax_payments[t] = accrued_tax
            accrued_tax = 0

    # 4. جدولة المقبوضات النقدية
    cash_collections = np.zeros(months)
    effective_credit_factor = 1 - data.bad_debt_rate
    
    for t in range(months):
        cash_collections[t] += revenue[t] * data.credit_terms[0]
        if t >= 1:
            cash_collections[t] += revenue[t-1] * data.credit_terms[1] * effective_credit_factor
        if t >= 2:
            cash_collections[t] += revenue[t-2] * data.credit_terms[2] * effective_credit_factor

    # 5. جدولة المدفوعات النقدية
    cash_outflows = np.zeros(months)
    for t in range(months):
        cash_outflows[t] += fixed_costs_adjusted[t]
        cash_outflows[t] += capex_outflows[t]
        cash_outflows[t] += tax_payments[t]
        
        if t >= data.cogs_payment_delay:
            cash_outflows[t] += variable_costs[t - data.cogs_payment_delay]
        else:
            cash_outflows[t] += variable_costs[t]

    # 6. مسار السيولة النقدية والتنبيهات
    beginning_cash = np.zeros(months)
    ending_cash = np.zeros(months)
    safety_alerts = []
    
    current_cash = data.initial_cash
    for t in range(months):
        beginning_cash[t] = current_cash
        net_flow = cash_collections[t] - cash_outflows[t]
        ending_cash[t] = current_cash + net_flow
        current_cash = ending_cash[t]
        
        if ending_cash[t] < data.min_safety_cash:
            safety_alerts.append(
                f"⚠️ تحذير سيولة في الشهر {t+1}: الرصيد المتوقع ({ending_cash[t]:,.0f}$) أقل من حد الأمان ({data.min_safety_cash:,.0f}$)"
            )

    # 7. حساب مؤشرات رأس المال العامل
    credit_sales_total = revenue.sum() * (1 - data.credit_terms[0])
    ar_ending = revenue[-1] * (data.credit_terms[1] + data.credit_terms[2])
    dso = (ar_ending / credit_sales_total) * (months * 30) if credit_sales_total > 0 else 0
    
    ap_ending = variable_costs[-1] if data.cogs_payment_delay > 0 else 0
    dpo = (ap_ending / variable_costs.sum()) * (months * 30) if variable_costs.sum() > 0 else 0

    # 8. تجهيز استجابة JSON للعميل
    financial_table = []
    for i in range(months):
        financial_table.append({
            "month": f"Month {i+1}",
            "revenue_accrual": round(float(revenue[i]), 2),
            "operating_costs": round(float(variable_costs[i] + fixed_costs_adjusted[i]), 2),
            "depreciation": round(float(depreciation[i]), 2),
            "ebt": round(float(ebt[i]), 2),
            "cash_inflows": round(float(cash_collections[i]), 2),
            "cash_outflows": round(float(cash_outflows[i]), 2),
            "capex": round(float(capex_outflows[i]), 2),
            "tax_paid": round(float(tax_payments[i]), 2),
            "net_cash_flow": round(float(cash_collections[i] - cash_outflows[i]), 2),
            "ending_cash": round(float(ending_cash[i]), 2)
        })

    return {
        "status": "success",
        "metrics": {
            "dso_days": round(float(dso), 1),
            "dpo_days": round(float(dpo), 1),
            "ccc_days": round(float(dso - dpo), 1),
            "safety_alerts": safety_alerts
        },
        "financial_schedule": financial_table
    }
