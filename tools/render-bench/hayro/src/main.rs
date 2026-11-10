use std::{sync::{Arc,Mutex},time::Instant,path::Path};
use rayon::prelude::*;
use serde_json::{json,Value};
use std::hash::{Hash,Hasher};
fn fingerprint(data:&[u8])->String {let mut h=std::collections::hash_map::DefaultHasher::new();data.hash(&mut h);format!("{:016x}",h.finish())}
// Retain each renderer's native owning pixmap: no artificial full-raster clone.
enum Image { Hayro(hayro::vello_cpu::Pixmap) }
impl Image {
    fn width(&self)->u32 {match self {Self::Hayro(p)=>p.width() as u32}}
    fn height(&self)->u32 {match self {Self::Hayro(p)=>p.height() as u32}}
    fn data(&self)->&[u8] {match self {Self::Hayro(p)=>bytemuck::cast_slice(p.data())}}
}
fn save(out:&str,p:usize,image:&Image)->anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let mut e=png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(Path::new(out).join(format!("{p:05}.png")))?),image.width(),image.height());
    e.set_color(png::ColorType::Rgba);e.set_depth(png::BitDepth::Eight);e.set_compression(png::Compression::Fast);
    e.write_header()?.write_image_data(image.data())?;Ok(())
}
fn execute<F:Fn(usize)->anyhow::Result<Image>+Sync>(count:usize,open_ms:f64,workers:usize,mode:&str,out:&str,indices:Vec<usize>,render:F)->anyhow::Result<()> {
    if mode=="sample" {
        let mut rows=Vec::new();
        for p in indices {for r in 0..26 {
            let t=Instant::now();let result=render(p);let ms=t.elapsed().as_secs_f64()*1000.;
            std::hint::black_box(&result);
            rows.push(json!({"page":p,"sample":r,"phase":if r==0 {"first"}else if r<6 {"warmup"}else{"warm"},"ms":ms,"ok":result.is_ok()}));
        }}
        println!("{}",json!({"open_ms":open_ms,"count":count,"rows":rows}));return Ok(());
    }
    let rows=Mutex::new(Vec::<Value>::new());
    let run=|p| {
        let t=Instant::now();let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||render(p)));let ms=t.elapsed().as_secs_f64()*1000.;
        let row=match result {
            Ok(Ok(image))=>{if mode=="dump" {save(out,p,&image).unwrap();}let mut row=json!({"page":p,"ms":ms,"ok":true,"width":image.width(),"height":image.height()});if mode=="verify"{row["hash"]=json!(fingerprint(image.data()));}row},
            Ok(Err(e))=>json!({"page":p,"ms":ms,"ok":false,"error":e.to_string()}),
            Err(_)=>json!({"page":p,"ms":ms,"ok":false,"error":"panic"})
        };rows.lock().unwrap().push(row);
    };
    let selected:Vec<usize>=std::env::args().nth(7).map(|s|s.split(',').filter_map(|s|s.parse().ok()).collect()).unwrap_or_else(||(0..count).collect());
    let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
    let t=Instant::now();
    if workers==1 {for &p in &selected {run(p);}}else{pool.install(||selected.par_iter().copied().for_each(run));}
    let wall_ms=t.elapsed().as_secs_f64()*1000.;let mut rows=rows.into_inner()?;rows.sort_by_key(|r|r["page"].as_u64());
    let ok=rows.iter().filter(|r|r["ok"]==true).count();
    println!("{}",json!({"open_ms":open_ms,"count":count,"selected_count":selected.len(),"ok":ok,"wall_ms":wall_ms,"rows":rows,"includes_png":mode=="dump"}));Ok(())
}
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect();let engine=&a[1];let file=&a[2];let scale:f32=a[3].parse()?;let workers:usize=a[4].parse()?;let mode=&a[5];let out=&a[6];
    let bytes=std::fs::read(file)?;
    let indices=|count:usize|a.get(7).map(|s|s.split(',').filter_map(|s|s.parse().ok()).collect()).unwrap_or(vec![0,count/2,count-1]);
    match engine.as_str() {
        "hayro"=>{
            use hayro::{hayro_syntax::Pdf,hayro_interpret::{InterpreterSettings,font::FontQuery,hayro_cmap::CidFamily}};
            // A child handles exactly one document, then exits. Keeping that document
            // until exit permits lifetime-correct worker-local caches without unsafe.
            let t=Instant::now();let pdf: &'static Pdf=Box::leak(Box::new(Pdf::new(bytes).map_err(|e|anyhow::anyhow!("{e:?}"))?));let count=pdf.pages().len();let open_ms=t.elapsed().as_secs_f64()*1000.;
            thread_local!{static HCACHE:hayro::RenderCache<'static>=hayro::RenderCache::new();}
            let is=InterpreterSettings{font_resolver:Arc::new(|query|match query {
                FontQuery::Standard(s)=>Some(s.get_font_data()),
                FontQuery::Fallback(f)=>{
                    let family=f.character_collection.as_ref().map(|c|c.family.clone());
                    let name=match family {Some(CidFamily::AdobeGB1|CidFamily::AdobeCNS1)=>Some("sc"),Some(CidFamily::AdobeJapan1)=>Some("jp"),Some(CidFamily::AdobeKorea1)=>Some("kr"),_=>None};
                    name.and_then(|n|std::fs::read(format!("{}/fonts/NotoSansCJK{n}-Regular.otf",env!("CARGO_MANIFEST_DIR"))).ok()).map(|v|(Arc::new(v) as hayro::hayro_interpret::font::FontData,0)).or_else(||Some(f.pick_standard_font().get_font_data()))
                }
            }),..Default::default()};
            execute(count,open_ms,workers,mode,out,indices(count),|p|{
                let pix=HCACHE.with(|cache|hayro::render(&pdf.pages()[p],cache,&is,&hayro::RenderSettings::default(),&hayro::PixmapSettings{x_scale:scale,y_scale:scale,bg_color:hayro::vello_cpu::color::palette::css::WHITE}));
                Ok(Image::Hayro(pix))
            })
        },
        _=>anyhow::bail!("unknown renderer")
    }
}
