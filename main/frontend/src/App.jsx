import MedicalDisclaimer from './components/MedicalDisclaimer';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';

function App() {
  return (
    <div className="min-h-screen flex flex-col bg-[#0f172a] text-slate-100">
      <MedicalDisclaimer />
      <Header />
      <Home />
      <Footer />
    </div>
  );
}

export default App;
