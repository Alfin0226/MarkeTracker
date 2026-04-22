"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { searchSymbols } from "@/lib/api";
import ProtectedRoute from "@/components/protected-route";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Search } from "lucide-react";

interface SearchResult {
  symbol: string;
  name: string;
}

function DashboardSearchContent() {
  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const router = useRouter();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (search.trim()) {
      router.push(`/dashboard/${search.trim().toUpperCase()}`);
    }
  };

  const handleSearchChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value.toUpperCase();
    setSearch(val);
    if (val.length > 1) {
      try {
        const data = await searchSymbols(val);
        setSuggestions(data);
      } catch {
        setSuggestions([]);
      }
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (sym: string) => {
    router.push(`/dashboard/${sym}`);
    setSearch("");
    setSuggestions([]);
  };

  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex items-center justify-center p-6">
      <Card className="w-full max-w-lg border-border/50 bg-card/50 backdrop-blur-sm shadow-2xl text-center">
        <CardHeader>
          <CardTitle className="text-xl font-bold">
            NASDAQ & NYSE Company Data Search
          </CardTitle>
          <p className="text-sm text-muted-foreground mt-1">
            Search by name or stock symbol.
          </p>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="relative">
            <Input
              type="text"
              placeholder="e.g., 'Apple' or 'AAPL'"
              value={search}
              onChange={handleSearchChange}
              autoComplete="off"
              className="text-center"
            />
            {suggestions.length > 0 && (
              <ul className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-hidden z-50 max-h-56 overflow-y-auto">
                {suggestions.map((s) => (
                  <li
                    key={s.symbol}
                    onClick={() => handleSuggestionClick(s.symbol)}
                    className="px-4 py-3 cursor-pointer text-sm hover:bg-accent transition-colors border-b border-border/30 last:border-0"
                  >
                    <strong className="text-violet-400">{s.symbol}</strong> -{" "}
                    {s.name}
                  </li>
                ))}
              </ul>
            )}
            <Button
              type="submit"
              className="mt-4 gap-2 bg-gradient-to-r from-violet-600 to-blue-600 text-white hover:from-violet-700 hover:to-blue-700"
            >
              <Search className="h-4 w-4" />
              Search
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

export default function DashboardSearchPage() {
  return (
    <ProtectedRoute>
      <DashboardSearchContent />
    </ProtectedRoute>
  );
}
