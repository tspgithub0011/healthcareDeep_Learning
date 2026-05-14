import { Activity, Github } from 'lucide-react';

/**
 * Glassmorphism header with logo + minimal nav.
 */
const Header = () => {
  return (
    <header className="sticky top-0 z-50 glass border-b border-slate-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <a href="/" className="flex items-center gap-2.5 group">
          <div className="p-2 rounded-lg bg-gradient-to-br from-primary to-secondary shadow-lg shadow-primary/20 group-hover:shadow-primary/40 transition-shadow duration-300">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-lg font-bold text-slate-100 tracking-tight">Healthcare</span>
            <span className="text-lg font-bold text-gradient">DL</span>
          </div>
        </a>

        {/* Nav */}
        <nav className="flex items-center gap-4">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors"
          >
            <Github className="w-4 h-4" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </nav>
      </div>
    </header>
  );
};

export default Header;
