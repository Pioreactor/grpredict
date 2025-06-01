## grpredict


> The study of the growth of bacterial cultures
does not constitute a specialized subject or branch
of research: it is the basic method of microbiology.”
Monitoring and controlling the specific growth rate
should not constitute a specialized subject or branch
of research: it should be the basic method of advanced
microbial bioprocessing.  
> \- Jacques Monod

The first step to controlling a culture's growth is to measure it. To do this, we need to model the rate of growth. We've seen other attempts at modelling the growth rate, but they all assumed too much:

 - Exponential growth model: perhaps truly early on, but certainly not the case near stationary.
 - Logistic growth model: requires an assumption about the max optical density, and assumes symmetric growth around the inflection point.
 - Gompertz models and other models: more parameters, less interpretable, and still make strong assumptions about the trajectory of growth.

Further more, none of these can model a culture in a turbidostat or chemostat mode, where the optical density is dropping quickly during a dilution, but the growth rate should remain constant.

### Re-visiting a growth model

We are often introduced to a growth model by the simple exponential growth model:

$\text{OD}(t) = \exp{ \text{gr} t  }$

Plainly put, the culture grows exponentially at rate $\text{gr}$. Like we mentioned above, this might be true for small time-scales, but certainly over the entire lag, log, and then stationary phases, this is not the case.

There's a hint of an interesting idea in the last paragraph though: _"over small time scales"_. What if we cut up the growth curve into many small time intervals, and computed a growth rate for each interval? Then our growth rate can be changing: starts near 0 in the lag phase, can increase to a peak in the log phase, and then drop to 0 again in the stationary phase. We also don't need to assume any parametric form for the growth rate, we can just measure it directly from the data.

Our new formula might look like:


$$\text{OD}(t) = \exp{ \left( \text{gr}_0 \Delta t  + \text{gr}_1 \Delta t + ...  \right)}$$

If we think more about this, and we keep shrinking our time interval towards zero, this is just an integral:

$$
\text{nOD}(t) = \exp{ \left( \int_0^t \text{gr}(s)ds \right)}
$$

There's still no particular assumption about the shape of the growth rate function, $gr(s)$. For example, consider the following ODs from a batch experiment:


![nod](https://github.com/user-attachments/assets/d0ed231c-d052-40bb-a1dd-0f3f0d14f811)

Using our estimation technique outlined below, we can estimate the growth curve as:

![gr](https://github.com/user-attachments/assets/5c1ce02e-9b60-40af-8016-9cc24ca8513f)


We can see that the growth is very dynamic, and certainly not a single number or a constrained form!


### Estimating a non-parametric growth rate

A non-parametric, dynamic growth rate sounds great, but we've replaced a estimation of a single value (or handle of values) to an entire function! This seems expensive!

Luckily, we do the work for you. We will compute $gr(t)$ using a statistical algorithm, **the Kalman filter**. The Kalman filter is an algorithm that estimates the state of a dynamic system from noisy measurements. In our case, the state is the growth rate, and the measurements are the optical density readings.  We input optical density observations one at a time, and the Kalman filter updates its estimate of the growth rate based on the new observation. This allows us to track the growth rate in real-time, as the culture grows.





