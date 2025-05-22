function secondsToHMS(secs) {
    secs = Math.max(0, parseInt(secs) || 0);
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    return `${h}h ${m}m ${s}s`;
}

async function fetchStatsAndDisplay() {
    try {
        const response = await fetch('http://127.0.0.1:8080/api/get_stats');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const data = await response.json();

        document.querySelector('.uptime').textContent = 'Uptime: ' + secondsToHMS(data.uptime);
        document.querySelector('.ver').textContent = 'Version: ' + (data.ver || '--');
        const now = new Date();
        document.querySelector('.updated').textContent = 'Last update: ' + now.toLocaleTimeString();

        const table = document.getElementById("stats-table");
        table.innerHTML = `
          <tr>
            <th>GPU</th>
            <th>Hashrate</th>
            <th>Temperature</th>
            <th>Fan</th>
            <th>Shares<br>(accepted/rejected)</th>
          </tr>
        `;

        function getShares(shares, i) {
            if (Array.isArray(shares?.[i])) {
                return `${shares[i][0] ?? '--'} / ${shares[i][1] ?? '--'}`;
            }
            if (i === 0 && Array.isArray(shares) && shares.length >= 2 && typeof shares[0] === "number") {
                return `${shares[0]} / ${shares[1]}`;
            }
            return "--";
        }

        const numGPU = Math.max(
          data.hs?.length || 0,
          data.temp?.length || 0,
          data.fan?.length || 0,
          (Array.isArray(data.shares) && Array.isArray(data.shares[0])) ? data.shares.length : 1
        );

        for(let i=0; i<numGPU; i++) {
            const hashrate = data.hs?.[i] ?? '--';
            const unit = data.hs_units ?? '';
            const temp = data.temp?.[i] ?? '--';
            const fan = data.fan?.[i] ?? '--';
            const sharesStr = getShares(data.shares, i);

            table.innerHTML += `
                <tr>
                  <td>GPU ${i}</td>
                  <td>${hashrate} ${unit}</td>
                  <td>${temp} °C</td>
                  <td>${fan} %</td>
                  <td>${sharesStr}</td>
                </tr>
            `;
        }
    } catch (error) {
        const table = document.getElementById("stats-table");
        table.innerHTML = `
          <tr>
            <th>GPU</th>
            <th>Hashrate</th>
            <th>Temperature</th>
            <th>Fan</th>
            <th>Shares<br>(accepted/rejected)</th>
          </tr>
          <tr>
            <td colspan="5">Error fetching stats</td>
          </tr>
        `;
        document.querySelector('.updated').textContent = 'Update error';
    }
}

fetchStatsAndDisplay();
setInterval(fetchStatsAndDisplay, 5000);
