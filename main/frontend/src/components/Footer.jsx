import { Heart, Shield } from 'lucide-react';

/**
 * Footer with credits, tech stack, and disclaimer repeat.
 */
const Footer = () => {
  return (
    <footer className="mt-auto border-t border-slate-800 bg-slate-950/60 py-8 px-4">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-4 text-center">
        {/* Tech stack badges */}
        <div className="flex flex-wrap items-center justify-center gap-2">
          {['FastAPI', 'PyTorch', 'React', 'Tailwind CSS', 'Vite'].map((tech) => (
            <span
              key={tech}
              className="px-2.5 py-1 text-[11px] font-medium rounded-full bg-slate-800/80 text-slate-400 border border-slate-700/50"
            >
              {tech}
            </span>
          ))}
        </div>

        {/* Built with line */}
        <p className="flex items-center gap-1.5 text-sm text-slate-500">
          Built with <Heart className="w-3.5 h-3.5 text-red-400 fill-red-400" /> for healthcare AI research
        </p>

        {/* Disclaimer */}
        <div className="flex items-start gap-2 max-w-lg">
          <Shield className="w-3.5 h-3.5 text-slate-600 mt-0.5 flex-shrink-0" />
          <p className="text-[11px] text-slate-600 leading-relaxed">
            This application is a demonstration of AI in medical imaging. It is not approved 
            by any regulatory body and must not be used for actual medical diagnosis.
          </p>
        </div>

        {/* Divider */}
        <div className="w-16 h-px bg-slate-800 my-1" />

        {/* Credits */}
        <div className="flex flex-col items-center gap-1.5">
          <p className="text-sm text-slate-400">
            Developed by <span className="text-slate-200 font-semibold">Tanmay Patil</span>
          </p>
          <a
            href="mailto:githubtsp0011@gmail.com"
            className="text-xs text-primary/70 hover:text-primary transition-colors"
          >
            githubtsp0011@gmail.com
          </a>
        </div>

        {/* Copyright */}
        <p className="text-[11px] text-slate-600">
          © {new Date().getFullYear()} Tanmay Patil. All rights reserved.
        </p>
      </div>
    </footer>
  );
};

export default Footer;
