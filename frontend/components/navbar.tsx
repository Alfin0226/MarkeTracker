"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/context/auth-context";
import { Button } from "@/components/ui/button";
import {
  BarChart3,
  Briefcase,
  Clock,
  Eye,
  LogOut,
  LogIn,
  UserPlus,
} from "lucide-react";

export default function Navbar() {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, logout } = useAuth();

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  const isActive = (path: string) => pathname === path;

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Brand */}
        <Link
          href="/"
          className="text-lg font-extrabold tracking-tight bg-gradient-to-r from-violet-500 to-blue-500 bg-clip-text text-transparent hover:opacity-90 transition-opacity"
        >
          MarkeTracker
        </Link>

        {/* Nav Links */}
        <div className="flex items-center gap-1">
          {isAuthenticated ? (
            <>
              <Link href="/dashboard">
                <Button
                  variant={isActive("/dashboard") ? "secondary" : "ghost"}
                  size="sm"
                  className="gap-1.5 text-sm font-medium"
                >
                  <BarChart3 className="h-4 w-4" />
                  <span className="hidden sm:inline">Dashboard</span>
                </Button>
              </Link>
              <Link href="/portfolio">
                <Button
                  variant={isActive("/portfolio") ? "secondary" : "ghost"}
                  size="sm"
                  className="gap-1.5 text-sm font-medium"
                >
                  <Briefcase className="h-4 w-4" />
                  <span className="hidden sm:inline">Portfolio</span>
                </Button>
              </Link>
              <Link href="/transactions">
                <Button
                  variant={isActive("/transactions") ? "secondary" : "ghost"}
                  size="sm"
                  className="gap-1.5 text-sm font-medium"
                >
                  <Clock className="h-4 w-4" />
                  <span className="hidden sm:inline">History</span>
                </Button>
              </Link>
              <Link href="/watchlist">
                <Button
                  variant={isActive("/watchlist") ? "secondary" : "ghost"}
                  size="sm"
                  className="gap-1.5 text-sm font-medium"
                >
                  <Eye className="h-4 w-4" />
                  <span className="hidden sm:inline">Watchlist</span>
                </Button>
              </Link>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleLogout}
                className="gap-1.5 text-sm font-medium text-muted-foreground hover:text-destructive"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Logout</span>
              </Button>
            </>
          ) : (
            <>
              <Link href="/login">
                <Button
                  variant={isActive("/login") ? "secondary" : "ghost"}
                  size="sm"
                  className="gap-1.5 text-sm font-medium"
                >
                  <LogIn className="h-4 w-4" />
                  Login
                </Button>
              </Link>
              <Link href="/register">
                <Button
                  size="sm"
                  className="gap-1.5 text-sm font-medium bg-gradient-to-r from-violet-600 to-blue-600 text-white hover:from-violet-700 hover:to-blue-700"
                >
                  <UserPlus className="h-4 w-4" />
                  Sign Up
                </Button>
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
