import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/auth-context";
import Navbar from "@/components/navbar";
import ErrorBoundary from "@/components/error-boundary";
import Script from "next/script";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MarkeTracker — Track. Analyze. Outperform.",
  description:
    "Real-time stock data, S&P 500 comparisons, and a $1M virtual portfolio to practice your trading strategy.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        {/* Google Analytics */}
        <Script
          strategy="afterInteractive"
          src={`https://www.googletagmanager.com/gtag/js?id=G-MK0NB67RVF`}
        />
        <Script
          id="google-analytics"
          strategy="afterInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('js', new Date());
              gtag('config', 'G-MK0NB67RVF');
            `,
          }}
        />
      </head>
      <body suppressHydrationWarning className={`${inter.variable} font-sans min-h-screen bg-background text-foreground antialiased`}>
        <AuthProvider>
          <Navbar />
          <ErrorBoundary>
            <main className="flex-1">{children}</main>
          </ErrorBoundary>
        </AuthProvider>
      </body>
    </html>
  );
}
