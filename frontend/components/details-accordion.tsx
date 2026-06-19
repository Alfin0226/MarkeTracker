"use client";

import { useState } from "react";
import { ChevronDown, ExternalLink } from "lucide-react";

interface IncomeGridItem {
  label: string;
  value: string;
  css_class: string;
}

interface StockData {
  longName?: string;
  sector?: string;
  industry?: string;
  marketCap?: string;
  website?: string;
  regularMarketDayLow?: number;
  regularMarketDayHigh?: number;
  fiftyTwoWeekLow?: number;
  fiftyTwoWeekHigh?: number;
  trailingPE?: number;
  trailingEps?: number;
  dividendYield?: number;
  averageAnalystRating?: string;
  targetMeanPrice?: number;
  forecast_price?: number;
  longBusinessSummary?: string;
  income_grid_items?: IncomeGridItem[];
}

interface DetailsAccordionProps {
  dashboardData: StockData;
  symbol: string;
}

function AccordionSection({
  id,
  title,
  isOpen,
  onToggle,
  children,
}: {
  id: string;
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="border-b border-border/40 last:border-b-0">
      <button
        onClick={onToggle}
        aria-expanded={isOpen}
        aria-controls={`panel-${id}`}
        className="w-full flex items-center justify-between py-4 px-1 text-left group"
      >
        <span className="text-sm font-semibold text-foreground">{title}</span>
        <ChevronDown
          className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>
      <div
        id={`panel-${id}`}
        role="region"
        aria-labelledby={id}
        className={`grid transition-[grid-template-rows] duration-200 ease-out ${
          isOpen ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
        }`}
      >
        <div className="overflow-hidden">
          <div className="pb-4 px-1">{children}</div>
        </div>
      </div>
    </div>
  );
}

export default function DetailsAccordion({
  dashboardData,
  symbol,
}: DetailsAccordionProps) {
  const [openSection, setOpenSection] = useState<string | null>(null);

  const toggle = (id: string) => {
    setOpenSection((prev) => (prev === id ? null : id));
  };

  return (
    <div className="mt-6 rounded-xl bg-card/30 ring-1 ring-foreground/5 px-4">
      {/* About this company */}
      <AccordionSection
        id="about"
        title="About this company"
        isOpen={openSection === "about"}
        onToggle={() => toggle("about")}
      >
        <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm mb-4">
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Sector
            </p>
            <p className="font-medium">{dashboardData.sector || "N/A"}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Industry
            </p>
            <p className="font-medium">{dashboardData.industry || "N/A"}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Market Cap
            </p>
            <p className="font-medium font-mono tabular-nums">
              {dashboardData.marketCap || "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Day Range
            </p>
            <p className="font-medium font-mono tabular-nums">
              ${dashboardData.regularMarketDayLow?.toFixed(2) || "N/A"} – $
              {dashboardData.regularMarketDayHigh?.toFixed(2) || "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              52-Week Range
            </p>
            <p className="font-medium font-mono tabular-nums">
              ${dashboardData.fiftyTwoWeekLow?.toFixed(2) || "N/A"} – $
              {dashboardData.fiftyTwoWeekHigh?.toFixed(2) || "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              P/E Ratio (TTM)
            </p>
            <p className="font-medium font-mono tabular-nums">
              {dashboardData.trailingPE?.toFixed(2) || "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              EPS (TTM)
            </p>
            <p className="font-medium font-mono tabular-nums">
              {dashboardData.trailingEps?.toFixed(2) || "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Dividend Yield
            </p>
            <p className="font-medium font-mono tabular-nums">
              {dashboardData.dividendYield
                ? `${dashboardData.dividendYield.toFixed(2)}%`
                : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Analyst Rating
            </p>
            <p className="font-medium capitalize">
              {dashboardData.averageAnalystRating || "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Target Price
            </p>
            <p className="font-medium font-mono tabular-nums">
              ${dashboardData.targetMeanPrice?.toFixed(2) || "N/A"}
            </p>
          </div>
        </div>

        {dashboardData.longBusinessSummary && (
          <p className="text-sm leading-relaxed text-foreground/80 mb-3">
            {dashboardData.longBusinessSummary}
          </p>
        )}

        {dashboardData.website && (
          <a
            href={dashboardData.website}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-emerald-500 hover:text-emerald-400 transition-colors"
          >
            Visit official site
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </AccordionSection>

      {/* Quarterly financials */}
      <AccordionSection
        id="financials"
        title="Quarterly financials"
        isOpen={openSection === "financials"}
        onToggle={() => toggle("financials")}
      >
        {dashboardData.income_grid_items &&
        Array.isArray(dashboardData.income_grid_items) &&
        dashboardData.income_grid_items.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {dashboardData.income_grid_items.map(
              (item: IncomeGridItem, idx: number) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border border-border/30 bg-background/30 ${
                    item.css_class === "positive"
                      ? "border-l-4 border-l-emerald-500"
                      : item.css_class === "negative"
                      ? "border-l-4 border-l-red-500"
                      : "border-l-4 border-l-muted-foreground/30"
                  }`}
                >
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                    {item.label}
                  </p>
                  <p
                    className={`text-base font-bold font-mono tabular-nums ${
                      item.css_class === "positive"
                        ? "text-emerald-500"
                        : item.css_class === "negative"
                        ? "text-red-500"
                        : "text-foreground"
                    }`}
                  >
                    {item.value}
                  </p>
                </div>
              )
            )}
          </div>
        ) : (
          <p className="text-muted-foreground text-sm py-2">
            Income statement data is not available for {symbol}.
          </p>
        )}
      </AccordionSection>

      {/* ML price forecast */}
      <AccordionSection
        id="forecast"
        title="ML price forecast"
        isOpen={openSection === "forecast"}
        onToggle={() => toggle("forecast")}
      >
        <div className="space-y-3">
          <div className="flex items-baseline gap-3">
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              Educational Trend Projection
            </span>
          </div>
          <p className="text-3xl font-bold font-mono tabular-nums text-foreground">
            {dashboardData.forecast_price
              ? `$${dashboardData.forecast_price.toFixed(2)}`
              : "Not Available"}
          </p>
          <div className="p-3 rounded-lg bg-muted/30 border border-border/30">
            <p className="text-xs text-muted-foreground leading-relaxed">
              <span className="font-semibold text-foreground/70">
                Disclaimer:
              </span>{" "}
              This is an educational trend projection, not a financial
              prediction. It is generated by a machine learning model for
              informational purposes only. Do not use this for real investment
              decisions. Past performance and model outputs do not guarantee
              future results.
            </p>
          </div>
        </div>
      </AccordionSection>
    </div>
  );
}
