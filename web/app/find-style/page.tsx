import Header from '../components/Header';
import Footer from '../components/Footer';
import ColorAnalysis from '../components/ColorAnalysis';

export const metadata = {
  title: 'Find Your Style — Wearhouse',
  description: 'Discover your color season through real-time face analysis.',
};

export default function FindStylePage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: '#ffe8d6' }}>
      <Header />

      <section
        className="relative overflow-hidden"
        style={{ paddingTop: '140px', paddingBottom: '0px', backgroundColor: '#ffe8d6' }}
      >
        {/* Decorative background bar */}
        <div
          className="absolute top-0 left-0 right-0 pointer-events-none"
          style={{
            height: '420px',
            background: 'linear-gradient(180deg, #ddbea9 0%, #ffe8d6 100%)',
            zIndex: 0,
          }}
        />

        <div className="relative z-10 px-8 md:px-16 max-w-5xl mx-auto">
          <span
            className="block text-xs uppercase mb-5"
            style={{ color: '#cb997e', letterSpacing: '0.35em' }}
          >
            Color Analysis
          </span>
          <h1
            className="font-light leading-none mb-6"
            style={{
              fontSize: 'clamp(2.5rem, 6vw, 5.5rem)',
              color: '#6b705c',
              fontFamily: "'Georgia', serif",
              lineHeight: '1.05',
            }}
          >
            Discover Your
            <br />
            <em style={{ color: '#cb997e', fontStyle: 'italic' }}>Color Season.</em>
          </h1>

          <div
            style={{
              height: '1px',
              width: '60px',
              backgroundColor: '#cb997e',
              marginBottom: '24px',
            }}
          />

          <p
            className="max-w-2xl mb-6"
            style={{
              color: '#6b705c',
              opacity: 0.75,
              lineHeight: '1.8',
              fontSize: '0.95rem',
            }}
          >
            Position your face in the frame. Our vision engine will read your skin, eye, and hair
            tones in real time, then translate them into the season that flatters you most — and
            the palette your wardrobe should live in.
          </p>
        </div>
      </section>

      <section
        className="relative px-8 md:px-16 pt-5 pb-24"
        style={{ backgroundColor: '#ffe8d6' }}
      >
        <ColorAnalysis />
      </section>

      <Footer />
    </div>
  );
}
